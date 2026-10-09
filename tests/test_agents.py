import subprocess
import unittest
from unittest.mock import Mock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents import RequestBudget, build_lead_agent, build_subagents
from sandbox import DockerSandbox
from tools import SOURCE_TOOLS, web_fetch


class AgentTests(unittest.TestCase):
    def test_shared_budget_runs_in_compiled_graph(self):
        class ToolCallingFakeModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""))
        backend = DockerSandbox(runner=runner, name="budget-test")
        model = ToolCallingFakeModel(messages=iter([AIMessage(content="Done")]))
        budget = RequestBudget(1)
        specs = build_subagents(budget)
        self.assertIs(specs[0]['middleware'][-1], budget)
        self.assertIs(specs[1]['middleware'][-1], budget)
        agent = build_lead_agent(backend, model, budget)
        agent.invoke({'messages': [{'role': 'user', 'content': 'hello'}]})
        self.assertEqual(budget.used, 1)
        with self.assertRaisesRegex(RuntimeError, 'Shared model request budget exhausted'):
            agent.invoke({'messages': [{'role': 'user', 'content': 'again'}]})

    def test_subagents_have_independent_limits_and_correct_source_tools(self):
        first, second = build_subagents(), build_subagents()
        self.assertEqual(first[0]["name"], "researcher")
        self.assertEqual(first[0]["tools"], SOURCE_TOOLS)
        self.assertEqual(first[1]["name"], "citation-checker")
        self.assertEqual(first[1]["tools"], [web_fetch])
        for spec in first:
            model_limit, tool_limit = spec["middleware"]
            self.assertEqual(model_limit.run_limit, 40)
            self.assertEqual(tool_limit.run_limit, 60)
        self.assertIsNot(first[0]["middleware"][0], first[1]["middleware"][0])
        self.assertIsNot(first[0]["middleware"][0], second[0]["middleware"][0])

    def test_compile_registers_planning_delegation_and_sandbox_tools(self):
        # Build a real graph without starting Docker or calling any model/API.
        class ToolCallingFakeModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        runner = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""))
        backend = DockerSandbox(runner=runner, name="compile-test")
        model = ToolCallingFakeModel(messages=iter([AIMessage(content="unused")]))
        graph = build_lead_agent(backend, model).get_graph()
        registered = graph.nodes["tools"].data.tools_by_name
        self.assertTrue({"task", "write_todos", "execute", "read_file", "write_file"} <= registered.keys())
        self.assertFalse({tool.name for tool in SOURCE_TOOLS} & registered.keys())
        # Compiling the graph must not execute commands in the sandbox.
        self.assertEqual(runner.call_count, 1)


if __name__ == "__main__":
    unittest.main()
