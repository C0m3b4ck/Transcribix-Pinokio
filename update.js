module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "venvs/{{args.group}}",
        path: ".",
        message: [
          "uv pip install --upgrade -r requirements/{{args.group}}.txt",
          "uv pip install --upgrade -r requirements/gradio.txt",
        ]
      }
    },
    {
      // torch/torchaudio are not in requirements/*.txt; refresh them from the
      // correct index so the CUDA install is never overwritten by a PyPI build
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "venvs/{{args.group}}",
          path: "."
        }
      }
    },
    {
      method: "input",
      params: {
        title: "Update Complete",
        description: "All dependencies have been updated."
      }
    },
  ]
}
