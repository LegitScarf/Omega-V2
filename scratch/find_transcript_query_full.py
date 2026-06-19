import json

transcript_path = r"C:\Users\KIIT\.gemini\antigravity\brain\a24ec535-33f6-4384-8428-b1326a400815\.system_generated\logs\transcript_full.jsonl"

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            # Check if any field contains the query
            line_str = json.dumps(data)
            if "awareness campaigns improve by 20%" in line_str:
                print("Found match in step_index:", data.get("step_index"))
                # Look for tool calls or reasoning content
                for call in data.get("tool_calls", []):
                    print("Tool call name:", call.get("name"))
                    print("Args:", call.get("args"))
                if "content" in data and data["content"]:
                    print("Content length:", len(data["content"]))
                    print("Snippet:", data["content"][:1000])
        except Exception as e:
            pass
