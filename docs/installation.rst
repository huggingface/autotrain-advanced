First steps with AutoNLP
===================================

Introduction
...................................

You can use AutoNLP in three ways:

#. With the AutoNLP `web interface <https://ui.autonlp.huggingface.co>`_
#. With the AutoNLP CLI tool
#. With the AutoNLP Python package

The AutoNLP UI is the recommended way to use AutoNLP if you're not familiar with Python or the terminal.
The CLI / Python package supports some additional tasks, and allows you to use AutoNLP programatically.

Installation instructions
...................................

Installing the AutoNLP Python package is super-easy! Since we are moving quite fast with the development of this library, it's always wise to use the latest version from pypi.

To install, you can use ``pip``:

.. code-block:: bash

    $ pip install -U autonlp


Please make sure that you have ``git-lfs`` installed. Check out the instructions here: https://github.com/git-lfs/git-lfs/wiki/Installation

First steps with the CLI
..................................

Logging in
----------------------------------

To log in AutoNLP, use your HuggingFace API token:

.. code-block:: bash

    autonlp login --api-key YOUR_HUGGING_FACE_API_TOKEN

Your API token can be found under in the settings section of your HuggingFace profile, on the `HuggingFace website <https://huggingface.co/settings/profile>`_

Using AutoNLP as an organization
----------------------------------

By default, AutoNLP creates and retrieves projects as you.
But you can create an AutoNLP project as an organization, if you are a member of this organization.
This will allow all the organization members to see and interact with the project.

To create an AutoNLP project as an organization, simply append: ``--username ORGANIZATION_NAME`` to the AutoNLP command.
For example, to retrieve a project's info that belongs to the organization ``my_org``, run:

.. code-block:: bash

    autonlp project_info --name my_project --username my_org

You can change the default identity AutoNLP uses to manage projects by running ``autonlp select_identity``:

.. code-block:: bash

    autonlp select_identity my_org

To print the current default identity and the ones available to you, run:

.. code-block:: bash

    autonlp list_identities
