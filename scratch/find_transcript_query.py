import json

transcript_path = r"C:\Users\KIIT\.gemini\antigravity\brain\a24ec535-33f6-4384-8428-b1326a400815\.system_generated\logs\transcript.jsonl"

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            content = str(data.get("content", ""))
            if "awareness campaigns improve by 20%" in content:
                print("Found match in step_index:", data.get("step_index"))
                # Print the content or tool call arguments
                tool_calls = data.get("tool_calls", [])
                if tool_calls:
                    print("Tool Calls:", json.dumps(tool_calls, indent=2))
                else:
                    print("Content:", content[:500])
        except Exception as e:
            pass
