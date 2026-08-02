module.exports = {
  run: [
    {
      // NVIDIA: install CUDA build of torch
      when: "{{gpu === 'nvidia'}}",
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
      // NVIDIA: verify the CUDA wheel imports correctly, emit a marker if it can't
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: [
          "python -c \"import torch\" 2>/dev/null || echo TORCH_IMPORT_FAILED"
        ],
        on: [{
          event: "/TORCH_IMPORT_FAILED/",
          done: true
        }]
      }
    },
    {
      // NVIDIA fallback: if the CUDA wheel can't import, install the CPU build
      when: "{{gpu === 'nvidia' && input.event}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: [
          "uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    },
    {
      // Non-NVIDIA: install CPU build of torch
      when: "{{gpu !== 'nvidia'}}",
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
