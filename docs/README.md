# Generating the documentation

To generate the documentation, you have to build it. Several packages are necessary to build the doc.

First, you need to install the project itself by running the following command at the root of the code repository:

```bash
pip install -e .
```

You also need to install 2 extra packages:

```bash
# `hf-doc-builder` to build the docs
pip install git+https://github.com/huggingface/doc-builder@main
# `watchdog` for live reloads
pip install watchdog
```

---
**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to commit the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages with the pip install command above,
you can generate the documentation by typing the following command:

```bash
doc-builder build autotrain docs/source/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

## Previewing the documentation

To preview the docs, run the following command:

```bash
doc-builder preview autotrain docs/source/
```

The docs will be viewable at [http://localhost:5173](http://localhost:5173). You can also preview the docs once you
have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

---
**NOTE**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update
`_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

---