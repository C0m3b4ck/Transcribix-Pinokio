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
      method: "input",
      params: {
        title: "Update Complete",
        description: "All dependencies have been updated."
      }
    },
  ]
}
