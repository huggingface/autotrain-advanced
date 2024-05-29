import collections

from huggingface_hub import list_models


def get_sorted_models(hub_models):
    hub_models = [{"id": m.modelId, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)
    hub_models = [m["id"] for m in hub_models]
    return hub_models


def _fetch_text_classification_models():
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


def _fetch_dreambooth_models():
    hub_models1 = list(
        list_models(
            task="text-to-image",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            filter=["diffusers:StableDiffusionXLPipeline"],
        )
    )
    hub_models2 = list(
        list_models(
            task="text-to-image",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            filter=["diffusers:StableDiffusionPipeline"],
        )
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models1 = list(
        list_models(
            task="text-to-image",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            filter=["diffusers:StableDiffusionXLPipeline"],
        )
    )
    trending_models2 = list(
        list_models(
            task="text-to-image",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            filter=["diffusers:StableDiffusionPipeline"],
        )
    )
    trending_models = list(trending_models1) + list(trending_models2)
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


def fetch_models():
    _mc = collections.defaultdict(list)
    _mc["text-classification"] = _fetch_text_classification_models()
    _mc["llm"] = _fetch_llm_models()
    _mc["image-classification"] = _fetch_image_classification_models()
    _mc["dreambooth"] = _fetch_dreambooth_models()
    _mc["seq2seq"] = _fetch_seq2seq_models()
    _mc["token-classification"] = _fetch_token_classification_models()
    _mc["text-regression"] = _fetch_text_classification_models()
    _mc["image-object-detection"] = _fetch_image_object_detection_models()
    _mc["sentence-transformers"] = _fetch_st_models()

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
