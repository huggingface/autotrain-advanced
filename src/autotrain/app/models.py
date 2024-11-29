import collections

from huggingface_hub import list_models


def get_sorted_models(hub_models):
    """
    Filters and sorts a list of models based on their download count.

    Args:
        hub_models (list): A list of model objects. Each model object must have the attributes 'id', 'downloads', and 'private'.

    Returns:
        list: A list of model IDs sorted by their download count in descending order. Only includes models that are not private.
    """
    hub_models = [{"id": m.id, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)
    hub_models = [m["id"] for m in hub_models]
    return hub_models


def _fetch_text_classification_models():
    """
    Fetches and sorts text classification models from the Hugging Face model hub.

    This function retrieves models for the tasks "fill-mask" and "text-classification"
    from the Hugging Face model hub, sorts them by the number of downloads, and combines
    them into a single list. Additionally, it fetches trending models based on the number
    of likes in the past 7 days, sorts them, and places them at the beginning of the list
    if they are not already included.

    Returns:
        list: A sorted list of model identifiers from the Hugging Face model hub.
    """
    hub_models1 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="text-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_llm_models():
    hub_models = list(
        list_models(
            task="text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)
    trending_models = list(
        list_models(
            task="text-generation",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_image_classification_models():
    hub_models = list(
        list_models(
            task="image-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="image-classification",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_image_object_detection_models():
    hub_models = list(
        list_models(
            task="object-detection",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            pipeline_tag="object-detection",
        )
    )
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="object-detection",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            pipeline_tag="object-detection",
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_seq2seq_models():
    hub_models = list(
        list_models(
            task="text2text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)
    trending_models = list(
        list_models(
            task="text2text-generation",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_token_classification_models():
    hub_models1 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="token-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_st_models():
    hub_models1 = list(
        list_models(
            task="sentence-similarity",
            library="sentence-transformers",
            sort="downloads",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=30,
            full=False,
        )
    )

    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="sentence-similarity",
            library="sentence-transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_vlm_models():
    hub_models1 = list(
        list_models(
            task="image-text-to-text",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            filter=["paligemma"],
        )
    )
    # hub_models2 = list(
    #     list_models(
    #         task="image-text-to-text",
    #         sort="downloads",
    #         direction=-1,
    #         limit=100,
    #         full=False,
    #         filter=["florence2"],
    #     )
    # )
    hub_models2 = []
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models1 = list(
        list_models(
            task="image-text-to-text",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            filter=["paligemma"],
        )
    )
    # trending_models2 = list(
    #     list_models(
    #         task="image-text-to-text",
    #         sort="likes7d",
    #         direction=-1,
    #         limit=30,
    #         full=False,
    #         filter=["florence2"],
    #     )
    # )
    trending_models2 = []
    trending_models = list(trending_models1) + list(trending_models2)
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def fetch_models():
    _mc = collections.defaultdict(list)
    _mc["text-classification"] = _fetch_text_classification_models()
    _mc["llm"] = _fetch_llm_models()
    _mc["image-classification"] = _fetch_image_classification_models()
    _mc["image-regression"] = _fetch_image_classification_models()
    _mc["seq2seq"] = _fetch_seq2seq_models()
    _mc["token-classification"] = _fetch_token_classification_models()
    _mc["text-regression"] = _fetch_text_classification_models()
    _mc["image-object-detection"] = _fetch_image_object_detection_models()
    _mc["sentence-transformers"] = _fetch_st_models()
    _mc["vlm"] = _fetch_vlm_models()
    _mc["extractive-qa"] = _fetch_text_classification_models()

    # tabular-classification
    _mc["tabular-classification"] = [
        "xgboost",
        "random_forest",
        "ridge",
        "logistic_regression",
        "svm",
        "extra_trees",
        "adaboost",
        "decision_tree",
        "knn",
    ]

    # tabular-regression
    _mc["tabular-regression"] = [
        "xgboost",
        "random_forest",
        "ridge",
        "svm",
        "extra_trees",
        "adaboost",
        "decision_tree",
        "knn",
    ]
    return _mc
