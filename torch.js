module.exports = {
  run: [
    {
      // Check if CUDA is available
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: [
          "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
        ],
        capture: {
          pattern: "/CUDA available: (True|False)/"
        }
      }
    },
    {
      // Install appropriate torch version based on CUDA availability
      when: "{{input.pattern[1] === 'True'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: [
          "uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
        ]
      }
    },
    {
      when: "{{input.pattern[1] === 'False'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: [
          "uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    },
  ]
}
