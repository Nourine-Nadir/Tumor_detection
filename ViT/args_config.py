PARSER_CONFIG = {
    "batch_size": {
        "flags": ("-bs", "--batch_size"),  # Can use proper tuple
        "help": "Batch size for training",
        "type": int,  # Direct Python type
        "default": 128
    },
    "epochs": {
        "flags": ("-ep", "--epochs"),
        "type": int,
        "help": "Number of epochs for training",
        "default": 10
    },
    "verbose": {
        "flags": ("-v", "--verbose"),
        "help": "Enable verbose output",
        "action": "store_true"
    }
}