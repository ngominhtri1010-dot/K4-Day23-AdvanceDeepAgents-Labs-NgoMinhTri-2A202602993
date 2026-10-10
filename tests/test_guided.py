import json
import subprocess
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents import RequestBudget, REPORT_PATH, SOURCES_PATH
from guided import GuidedRun, build_guided_agent, parse_web, relevant
from sandbox import DockerSandbox


class PhaseModel(BaseChatModel):
    chosen: str = ''

    @property
    def _llm_type(self):
        return 'phase-test'

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={'chosen': tools[0].name})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        data = json.loads(messages[-1].content.rsplit('\n/no_think', 1)[0])
        if 'You are a researcher' in system:
            result = AIMessage(content='Retrieved sources describe complementary methods and limitations.')
        elif 'You are a citation checker' in system:
            result = AIMessage(content=json.dumps({'checks': [
                {'n': p['n'], 'status': 'SUPPORTED', 'quote_id': p['evidence_quotes'][0]['quote_id']} for p in data['claims_and_pages']]}))
        elif self.chosen == 'plan_research':
            result = AIMessage(content='', tool_calls=[{'name': self.chosen, 'id': 'plan', 'args': {
                'title': 'Synthetic test survey', 'questions': [
                    {'question': f'Question {i}', 'query': 'test topic', 'heading': f'Theme {i}'} for i in range(3)]}}])
        else:
            catalog = data['catalog']
            first = {r['source']: r['n'] for r in catalog}
            if data['section'] == 'TL;DR':
                text = '\n'.join(f'- Retrieved evidence supports the described method [{n}].' for n in first.values())
            else:
                text = 'Retrieved evidence describes complementary methods ' + ''.join(f'[{n}]' for n in first.values()) + '.'
            result = AIMessage(content='', tool_calls=[{'name': self.chosen, 'id': 'write-' + data['section'], 'args': {'markdown': text}}])
        return ChatResult(generations=[ChatGeneration(message=result)])


class GuidedTests(unittest.TestCase):
    def test_web_metadata_is_only_taken_from_search_blocks(self):
        records = parse_web('Title: Actual title\nURL: https://example.org/paper\nPublished: N/A\nHighlights:\nEvidence\nhttps://fake.example/inside-body\n')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['url'], 'https://example.org/paper')
        self.assertEqual(records[0]['date'], '')
        self.assertEqual(parse_web('ERROR: no sources'), [])

    def test_world_model_relevance_rejects_keyword_noise(self):
        topic = 'survey about world model'
        self.assertFalse(relevant({'title': 'Forecasting Imports', 'evidence': 'Neural models forecast world economic imports.'}, topic))
        self.assertFalse(relevant({'title': 'Uncertain Combinatorics', 'evidence': 'World uncertainty modeled with fuzzy sets and graphs.'}, topic))
        self.assertTrue(relevant({'title': 'World Models', 'evidence': 'Learned generative dynamics simulate reinforcement learning environments.'}, topic))
        self.assertTrue(relevant({'title': 'DreamTrue', 'evidence': 'A robot world model predicts physically plausible video.'}, topic))

    def test_agent_relevance_distinguishes_react_namesakes(self):
        topic = 'survey about LLM agents and tool use'
        self.assertFalse(relevant({'title': 'REACT: Rolling Denoising for Reactive Robot Control', 'evidence': 'Vision-language models generate action chunks for robot motion.'}, topic))
        self.assertFalse(relevant({'title': 'Reactive Transformer', 'evidence': 'Event-driven reactive language models provide low latency.'}, topic))
        self.assertTrue(relevant({'title': 'ReAct: Synergizing Reasoning and Acting in Language Models', 'evidence': 'Reasoning traces interleave with actions.'}, topic))
        self.assertTrue(relevant({'title': 'ACEBench', 'evidence': 'Evaluate tool usage in large language models.'}, topic))

    def test_video_relevance_rejects_transport_and_perception_only(self):
        topic = 'survey about video and multimodal generation'
        self.assertFalse(relevant({'title': 'E2E-WAVE: Waveform Generation for Video Multicasting', 'evidence': 'Semantic waveforms transport underwater video.'}, topic))
        self.assertFalse(relevant({'title': 'FastBench', 'evidence': 'Streaming VLMs perceive high-dynamic video streams.'}, topic))
        self.assertTrue(relevant({'title': 'Movie Gen', 'evidence': 'Video generation with synchronized audio.'}, topic))
        self.assertTrue(relevant({'title': 'OuroWorld', 'evidence': 'Diverse looping 3D cinemagraphs.'}, topic))

    def test_inference_relevance_rejects_incidental_model_mentions(self):
        topic = 'survey about efficient inference and small language models'
        self.assertFalse(relevant({'title': 'Anomaly Detection in Shell Sessions', 'evidence': 'We use DistilBERT, a compressed language model.'}, topic))
        self.assertFalse(relevant({'title': 'Purifying Backdoored Models', 'evidence': 'Efficient inference defends large language models.'}, topic))
        self.assertTrue(relevant({'title': 'DistilBERT: a distilled version of BERT, smaller and faster', 'evidence': 'Transformer distillation.'}, topic))
        self.assertTrue(relevant({'title': 'Scaling Inference-Efficient Language Models', 'evidence': 'Architecture affects latency.'}, topic))

    def test_real_graph_dispatches_tasks_and_locks_metadata(self):
        files = {}
        def execute(args, **kwargs):
            if any('finalize_citations.py' in str(arg) for arg in args) and REPORT_PATH in files:
                report = files[REPORT_PATH].decode().split('\n## References', 1)[0]
                sources = json.loads(files[SOURCES_PATH])
                files[REPORT_PATH] = (report + '\n## References\n' + ''.join(
                    f"[{r['n']}] {r['title']}. {r['url']}\n" for r in sources)).encode()
            return subprocess.CompletedProcess([], 0, stdout=b'OK', stderr=b'')
        runner = Mock(side_effect=execute)
        backend = DockerSandbox(runner=runner, name='guided-test')
        hf = json.dumps([{'id': '2601.00001', 'url': 'https://huggingface.co/papers/2601.00001', 'title': 'Actual HF title', 'summary': 'Evidence on methods', 'published': '2020-01-01'}])
        arxiv = json.dumps([{'id': '2601.00002', 'url': 'https://arxiv.org/abs/2601.00002', 'title': 'Actual arXiv title', 'summary': 'Evidence on methods', 'published': '2026-01-01'}])
        web = 'Title: Actual project\nURL: https://example.org/project\nPublished: 2026-01-01\nHighlights:\nEvidence on methods'
        def upload(_, entries):
            files.update(entries)
        def download(_, paths):
            return {p: files[p] for p in paths if p in files}
        with patch('guided.upload', side_effect=upload), patch('guided.download', side_effect=download), \
             patch('guided.hf_search_papers', SimpleNamespace(name='hf_search_papers', invoke=Mock(return_value=hf))), \
             patch('guided.arxiv_search', SimpleNamespace(name='arxiv_search', invoke=Mock(return_value=arxiv))), \
             patch('guided.web_search', SimpleNamespace(name='web_search', invoke=Mock(return_value=web))), \
             patch('guided.web_fetch', SimpleNamespace(name='web_fetch', invoke=Mock(return_value='Retrieved evidence supports the described method and explains limits.'))):
            budget = RequestBudget(20)
            agent, run = build_guided_agent(backend, PhaseModel(), 'test topic', budget)
            result = agent.invoke({'messages': [{'role': 'user', 'content': 'test topic'}]}, {'recursion_limit': 100})
            self.assertEqual(len(run.jobs), 3)
            self.assertEqual(run.counter['task'], 4)  # Three researchers plus the checker, actually executed.
            self.assertEqual(budget.used, 11)  # Synthetic coordinator actions do not count as model requests.
            self.assertEqual(run.counter['write_report_section'], 6)
            sources = json.loads(files[SOURCES_PATH])
            self.assertEqual({r['source'] for r in sources}, {'arxiv', 'hf-search', 'web'})
            self.assertEqual({r['title'] for r in sources}, {'Actual HF title', 'Actual arXiv title', 'Actual project'})
            self.assertTrue(all(c['status'] == 'SUPPORTED' for c in run.audit['checks']))
            self.assertTrue(all(c['evidence'] == 'Retrieved evidence supports the described method and explains limits.' for c in run.audit['checks']))
            self.assertIn(b'## Trends and open problems', files[REPORT_PATH])
            writer = next(t for t in run.tools if t.name == 'write_report_section')
            run.sections.pop('Background')
            before = files[REPORT_PATH]
            self.assertTrue(writer.invoke({'markdown': 'Invented claim [999].'}).startswith('ERROR:'))
            self.assertEqual(files[REPORT_PATH], before)
            run.persist()
            restored = {path: raw.decode() for path, raw in files.items()}
            next_budget = RequestBudget(8)
            next_agent, next_run = build_guided_agent(backend, PhaseModel(), 'test topic', next_budget, restored)
            next_agent.invoke({'messages': [{'role': 'user', 'content': 'resume'}]}, {'recursion_limit': 100})
            self.assertEqual(next_budget.used, 2)  # One missing section plus one real checker call.
            self.assertEqual(next_run.counter['hf_search_papers'], 3)  # No repeat discovery.
            self.assertEqual(next_run.counter['task'], 5)  # Original three researchers and two actual checks.



if __name__ == '__main__':
    unittest.main()
