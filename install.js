module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "local.set",
      params: {
        state: {
          group: "{{args.group}}"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "sudo apt install -y libcublas12",
        ],
      }
    },
    // Whisper Variants
    {
      when: "{{args.group === 'whisper'}}",
      method: "shell.run",
      params: {
        venv: "venvs/whisper",
        path: ".",
        message: [
          "uv pip install -r requirements/whisper.txt",
          "uv pip install -r requirements/gradio.txt",
        ],
      }
    },
    // NVIDIA Models
    {
      when: "{{args.group === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "venvs/nvidia",
        path: ".",
        message: [
          "uv pip install -r requirements/nvidia.txt",
          "uv pip install -r requirements/gradio.txt",
        ],
      }
    },
    // Lightweight Models
    {
      when: "{{args.group === 'lightweight'}}",
      method: "shell.run",
      params: {
        venv: "venvs/lightweight",
        path: ".",
        message: [
          "uv pip install -r requirements/lightweight.txt",
          "uv pip install -r requirements/gradio.txt",
        ],
      }
    },
    // All Models
    {
      when: "{{args.group === 'all'}}",
      method: "shell.run",
      params: {
        venv: "venvs/all",
        path: ".",
        message: [
          "uv pip install -r requirements/all.txt",
          "uv pip install -r requirements/gradio.txt",
        ],
      }
    },
    // Install torch for the selected group
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "venvs/{{args.group}}",
          path: "."
        }
      }
    },
    // Post-install message
    {
      method: "input",
      params: {
        title: "Install Complete",
        description: "Installation complete. You may now start the app."
      }
    },
  ]
}
