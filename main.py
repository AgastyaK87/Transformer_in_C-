import json

def create_dataset_from_nl2bash(nl_path, cm_path, output_path):
    """
    Reads the nl2bash dataset and converts it into the JSONL format
    required for the OS-LLM Transformer.
    """
    print(f"Reading prompts from: {nl_path}")
    print(f"Reading commands from: {cm_path}")

    with open(nl_path, 'r', encoding='utf-8') as nl_file, \
            open(cm_path, 'r', encoding='utf-8') as cm_file, \
            open(output_path, 'w', encoding='utf-8') as out_file:

        prompts = nl_file.readlines()
        commands = cm_file.readlines()

        if len(prompts) != len(commands):
            print("Error: The number of lines in the prompt and command files do not match.")
            return

        print(f"Found {len(prompts)} total prompt/command pairs.")

        processed_count = 0
        for i in range(len(prompts)):
            prompt = prompts[i].strip()
            command = commands[i].strip()

            if not prompt or not command:
                continue

            # This logic determines if the command is a simple shell command
            # or a multi-line script that should be handled by 'create_script'.
            if command.startswith("#!") or "\n" in command:
                action = {
                    "type": "create_script",
                    "filename": "script.sh", # A placeholder filename
                    "language": "bash",
                    "content": command
                }
            else:
                action = {
                    "type": "execute_shell",
                    "command": command
                }

            # This is the final JSON structure for one training example
            json_record = {
                "prompt": prompt,
                "plan": {
                    "actions": [action]
                }
            }

            # Write the JSON object as a single line in the output file
            out_file.write(json.dumps(json_record) + "\n")
            processed_count += 1

    print(f"Successfully processed and wrote {processed_count} records to {output_path}")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Update this path to point to your cloned nl2bash repository directory
    NL2BASH_DATA_DIR = './nl2bash/data/bash/'

    PROMPT_FILE = NL2BASH_DATA_DIR + 'all.nl'
    COMMAND_FILE = NL2BASH_DATA_DIR + 'all.cm'
    OUTPUT_FILE = './dataset.jsonl'

    create_dataset_from_nl2bash(PROMPT_FILE, COMMAND_FILE, OUTPUT_FILE)