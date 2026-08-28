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
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps"
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
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
        ]
      }
    },
  ]
}