import unittest

from datasets import tasks

from autonlp.evaluate import format_datasets_task, get_compatible_task_template


class TestFormatDatasetsTask(unittest.TestCase):
    def test_binary_classification(self):
        autonlp_task = "binary_classification"
        formatted_task = format_datasets_task("text_classification", "allocine")
        self.assertEqual(autonlp_task, formatted_task)

    def test_multi_class_classification(self):
        autonlp_task = "multi_class_classification"
        formatted_task = format_datasets_task("text_classification", "ag_news")
        self.assertEqual(autonlp_task, formatted_task)

    def test_dataset_without_template(self):
        # TODO(lewtun): Replace with dedicated dummy dataset on Hugging Face Hub
        self.assertRaises(
            ValueError,
            format_datasets_task,
            "text_classification",
            "patrickvonplaten/scientific_papers_dummy",
            "pubmed",
        )


class TestGetCompatibleTaskTemplate(unittest.TestCase):
    def test_text_classification(self):
        task_template = get_compatible_task_template("text_classification", "allocine")
        self.assertIsInstance(task_template, tasks.TextClassification)

    def test_dataset_without_template(self):
        # TODO(lewtun): Replace with dedicated dummy dataset on Hugging Face Hub
        task_template = get_compatible_task_template(
            "text_classification", "patrickvonplaten/scientific_papers_dummy", "pubmed"
        )
        self.assertIsNone(task_template)

    def test_incompatible_task(self):
        self.assertRaises(ValueError, get_compatible_task_template, "text_classification", "squad")
