# Guidelines for Contributing

As a scientific community-driven software project, ArviZ.jl welcomes contributions from interested individuals or groups. These guidelines are provided to give potential contributors information to make their contribution compliant with the conventions of the ArviZ.jl project, and maximize the probability of such contributions to be merged as quickly and efficiently as possible.

There are 4 main ways of contributing to the ArviZ.jl project (in descending order of difficulty or scope):

* Adding new or improved functionality to the existing codebase
* Fixing outstanding issues (bugs) with the existing codebase. They range from low-level software bugs to higher-level design problems.
* Contributing or improving the documentation (`docs`) or examples
* Submitting issues related to bugs or desired enhancements

# Opening issues

We appreciate being notified of problems with the existing ArviZ.jl code. We prefer that issues be filed the on [Github Issue Tracker](https://github.com/arviz-devs/ArviZ.jl/issues), rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are
inclined to do so to submit patches for new or existing issues via pull
requests. This is particularly the case for simple fixes, such as typos
or tweaks to documentation, which do not require a heavy investment
of time and attention.

Contributors are also encouraged to contribute new code to enhance ArviZ.jl's
functionality, also via pull requests.
Please consult the [ArviZ.jl documentation](https://arviz-devs.github.io/ArviZ.jl/)
to ensure that any new contribution does not strongly overlap with existing functionality.

In general, new analysis tools such as plots, statistics, and diagnostics
should be contributed directly to [ArviZ](https://arviz-devs.github.io/arviz/).
Julia-specific tools such as improvements to the API, documentation, custom
types, conversion functions, and interfaces with other Julia packages should
be added to ArviZ.jl.

The preferred workflow for contributing to ArviZ.jl is to fork
the [GitHub repository](https://github.com/arviz-devs/ArviZ.jl/), clone it to your local machine, and develop on a feature branch.

For more instructions see the
[Pull request checklist](#pull-request-checklist)

### Consistency with ArviZ
To facilitate each transfer between platforms, ArviZ.jl should as much as
possible keep the same interface, behavior, and style for official API
functions as [ArviZ](https://arviz-devs.github.io).

### Code Formatting
For code generally follow the
[Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/index.html)
and the [Blue Style Guide](https://github.com/invenia/BlueStyle), the latter
taking precedence.

Before submission, final formatting should be done with
[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl).
For more detailed steps on a typical development workflow see the
[Pull request checklist](#pull-request-checklist)

### Docstring formatting
Functions intended for use by a user must be documented and should follow the
[Blue documentation guide](https://github.com/invenia/BlueStyle#documentation)
Please reasonably document any additions or changes to the codebase,
when in doubt, add a docstring.

## Steps

1. Fork the [project repository](https://github.com/arviz-devs/ArviZ.jl/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the ArviZ.jl repo from your GitHub account to your local disk and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/ArviZ.jl.git
   $ cd ArviZ.jl
   $ git remote add upstream git@github.com:arviz-devs/ArviZ.jl.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``master`` branch of any repository.

4. Package dependencies are in ``Project.toml``. To set up a development
   environment, in the Julia REPL run:

   ```julia
   ] dev .
   ```

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the ArviZ.jl repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

## Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* All public methods must have informative docstrings with sample usage when appropriate.

* Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

* All other tests pass when everything is rebuilt from scratch.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Documentation follows Blue style guide

* Code coverage **cannot** decrease. Coverage is automatically checked on all pull requests

* Your code has been formatted with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) with default settings. From the REPL:

  ```julia
  julia> using JuliaFormatter
  julia> format("src/")
  ```

#### This guide was derived from the [ArviZ guidelines for contributing](https://github.com/arviz-devs/arviz/blob/master/CONTRIBUTING.md)
