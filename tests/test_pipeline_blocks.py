import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline_blocks import UseCodeToolBlock, SynthesizeFinalAnswerBlock
from src.thinking_pipeline import ThinkingPipeline


@pytest.fixture
def code_tool_block():
    return UseCodeToolBlock()


def test_should_generate_plot_detects_keywords(code_tool_block):
    assert code_tool_block._should_generate_plot("Please graph the sales trend", None) is True
    assert code_tool_block._should_generate_plot("Summarize the dataset", "" ) is False
    assert code_tool_block._should_generate_plot("Compute results", "Create a line chart of the output") is True


def test_parse_structured_output_json_only(code_tool_block):
    payload = code_tool_block._parse_structured_output('{"answer": "ok", "details": {"value": 42}}')
    assert payload == {"answer": "ok", "details": {"value": 42}}


def test_parse_structured_output_with_noise(code_tool_block):
    stdout = 'intermediate log\n{"answer": "done", "details": {"table": [1, 2, 3]}}'
    payload = code_tool_block._parse_structured_output(stdout)
    assert payload == {"answer": "done", "details": {"table": [1, 2, 3]}}


def test_ensure_required_imports_adds_matplotlib(code_tool_block):
    code = "plt.plot([1, 2, 3], [1, 4, 9])\nplt.show()"
    new_code, fixes = code_tool_block._ensure_required_imports(code)
    assert "import matplotlib.pyplot as plt" in new_code
    assert any("matplotlib" in fix for fix in fixes)


def test_collate_collected_info_tracks_assets():
    block = SynthesizeFinalAnswerBlock()
    collected = {
        "use_code_tool_0": {
            "output": "{\"answer\": \"ok\"}",
            "plots": ["/tmp/plot.png"],
            "structured_output": {"answer": "ok", "details": {"value": 7}},
        },
        "plan_creation_0": "Plan text",
    }
    collated, assets = block._collate_collected_info(collected)

    assert assets == ["/tmp/plot.png"]
    assert collated["blocks"]["use_code_tool_0"]["structured_output"] == {"answer": "ok", "details": {"value": 7}}
    assert collated["blocks"]["plan_creation_0"] == "Plan text"


def test_thinking_pipeline_tracks_assets():
    pipeline = ThinkingPipeline(verbose=False)
    pipeline.reset_state("test prompt")

    class DummyCodeBlock:
        def __call__(self, *_args, **_kwargs):
            return {
                "output": "{}",
                "plots": ["/tmp/generated.png"],
                "structured_output": {"answer": "ok", "details": {}},
            }

    class DummySynthesisBlock:
        def __call__(self, prompt, plan, context):
            return {"final_response": "done"}

    pipeline.block_registry["use_code_tool"] = DummyCodeBlock()
    pipeline.block_registry["synthesize_final_answer"] = DummySynthesisBlock()

    spec = [
        {"key": "use_code_tool", "data": {}},
        {"key": "synthesize_final_answer", "data": {}},
    ]

    pipeline.execute_pipeline(spec)

    assert "/tmp/generated.png" in pipeline.state["assets"]
