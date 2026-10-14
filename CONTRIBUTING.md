# Contributing to Cafe Analyzer

First off, thanks for taking the time to contribute! Such projects are built by the community.

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a Pull Request.

## Getting Started
*   The project uses **YOLOv8x** for detection and **OpenAI CLIP** for advanced gender classification.
*   You will need a Google Colab environment or a local machine with a strong GPU (NVIDIA A100, V100 or T4 recommended for Colab).
*   All dependencies are listed in the initial setup cells of the notebook.

## How to Report a Bug?

If you find a bug, please create an issue in the GitHub repository.
Include as much detail as possible:
*   **Use a clear and descriptive title.**
*   **Describe the exact steps to reproduce the bug.**
*   **Provide specific examples or screenshots.**
*   **Include details of your environment (OS, GPU, Python version, Libraries versions).**

## How to Suggest an Enhancement?

If you have an idea to make the project better, please open an issue with the tag "enhancement".
*   Explain why the enhancement would be useful to the majority of users.
*   Provide as much detail and code examples as possible.

## Your First Contribution

Unsure where to begin? You can start by:
*   **Tackling "good first issue" tags** (if available).
*   **Improving documentation:** Typos, clarifications, or adding examples.
*   **Adding support for new video formats.**

## Pull Request Process

1.  Fork the repository.
2.  Create your feature branch.
3.  Test your changes thoroughly in your Colab environment.
4.  Commit your changes with a clear commit message.
5.  Push to your fork and submit a Pull Request.
6.  Maintainers will review your code. Please be patient.

## Style Guidelines
*   This project follows **PEP 8** for Python code.
*   Please use clear comments, especially when integrating new models or changing counting logic.
*   Ensure that your changes do not decrease the overall FPS significantly on standard Colab GPUs.

---

Thank you,
Emirhan Gökçe
