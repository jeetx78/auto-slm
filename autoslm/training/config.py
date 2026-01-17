def training_config_for_task(task: str):
    """
    Decide training hyperparameters based on inferred task
    """

    if task == "qa":
        return {
            "prompt_style": "qa",
            "epochs": 2,
            "learning_rate": 2e-4,
            "max_seq_len": 512,
            "eval_metric": "exact_match"
        }

    if task == "summarization":
        return {
            "prompt_style": "summarize",
            "epochs": 1,
            "learning_rate": 1e-4,
            "max_seq_len": 1024,
            "eval_metric": "rouge"
        }

    if task == "extraction":
        return {
            "prompt_style": "extract",
            "epochs": 3,
            "learning_rate": 3e-4,
            "max_seq_len": 512,
            "eval_metric": "f1"
        }

    return {
        "prompt_style": "assistant",
        "epochs": 1,
        "learning_rate": 1e-4,
        "max_seq_len": 512,
        "eval_metric": "none"
    }
