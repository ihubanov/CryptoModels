MODEL_TO_HASH = {
    "qwen3-embedding-0.6b": "bafkreiacd5mwy4a5wkdmvxsk42nsupes5uf4q3dm52k36mvbhgdrez422y",
    "qwen3-embedding-4b": "bafkreia7nzedkxlr6tebfxvo552zq7cba6sncloxwyivfl3tpj7hl5dz5u",
    "qwen3-1.7b": "bafkreib6pws5dx5ur6exbhulmf35twfcizdkxvup4cklzprlvaervfz5zy",
    "qwen3-4b": "bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza",
    "qwen3-8b": "bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy",
    "qwen3-14b": "bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm",
    "qwen3-32b": "bafkreihq4usl2t3i6pqoilvorp4up263yieuxcqs6xznlmrig365bvww5i",
    "qwen3-30b-a3b": "bafkreieroiopteqmtbjadlnpq3qkakdu7omvtuavs2l2qbu46ijnfdo2ly",
    "gemma-3-4b": "bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q",
    "gemma-3-12b": "bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq",
    "gemma-3-27b": "bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve",
    "gemma-3n-e4b": "bafkreihz3mz422vpoy7sccwj5tujkerxbjxdlmsqqf3ridxbe3m6ipnq5i",
    "lfm2-1.2b": "bafkreiaztifhss23cftya3bkorbsenzohol2oc3dvngo2srbbosko6gmme",
    "devstral-small": "bafkreih4xgr5t7yc3yooz6i6usgpwhggaobspmgut4rnu42gi6cv77o4em",
    "openreasoning-nemotron-32b": "bafkreidrdplo7mcfhrvocaa26yge6kmxmuwrexm5rffnzo5lbe6fkhjuvq",
    "flux-dev": "bafkreiaha3sjfmv4affmi5kbu6bnayenf2avwafp3cthhar3latmfi632u",
    "flux-schnell": "bafkreibks5pmc777snbo7dwk26sympe2o24tpqfedjq6gmgghwwu7iio34",
    "flux-dev-nsfw": "bafkreidbaksrogxispjejczfj36vtf5uzsbjt7irspl6kckynz5u2ugzke"
}

MODELS = {
    "bafkreiacd5mwy4a5wkdmvxsk42nsupes5uf4q3dm52k36mvbhgdrez422y": {
        "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "model": "Qwen3-Embedding-0.6B-Q8_0.gguf"
    },
    "bafkreia7nzedkxlr6tebfxvo552zq7cba6sncloxwyivfl3tpj7hl5dz5u": {
        "repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "model": "Qwen3-Embedding-4B-Q8_0.gguf"
    },
    "bafkreib6pws5dx5ur6exbhulmf35twfcizdkxvup4cklzprlvaervfz5zy": {
        "repo": "Qwen/Qwen3-1.7B-GGUF",
        "model": "Qwen3-1.7B-Q8_0.gguf"
    },
    "bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza": {
        "repo": "Qwen/Qwen3-4B-GGUF",
        "model": "Qwen3-4B-Q8_0.gguf"
    },
    "bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "model": "Qwen3-8B-Q8_0.gguf"
    },
    "bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm": {
        "repo": "Qwen/Qwen3-14B-GGUF",
        "model": "Qwen3-14B-Q8_0.gguf"
    },
    "bafkreihq4usl2t3i6pqoilvorp4up263yieuxcqs6xznlmrig365bvww5i": {
        "repo": "Qwen/Qwen3-32B-GGUF",
        "model": "Qwen3-32B-Q8_0.gguf"
    },
    "bafkreieroiopteqmtbjadlnpq3qkakdu7omvtuavs2l2qbu46ijnfdo2ly": {
        "repo": "Qwen/Qwen3-30B-GGUF",
        "model": "Qwen3-30B-A3B-Q8_0.gguf"
    },
    "bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q": {
        "repo": "lmstudio-community/gemma-3-4B-it-qat-GGUF",
        "model": "gemma-3-4B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf"
    },
    "bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq": {
        "repo": "lmstudio-community/gemma-3-12B-it-qat-GGUF",
        "model": "gemma-3-12B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf"
    },
    "bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve": {
        "repo": "lmstudio-community/gemma-3-27B-it-qat-GGUF",
        "model": "gemma-3-27B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf"
    },
    "bafkreihz3mz422vpoy7sccwj5tujkerxbjxdlmsqqf3ridxbe3m6ipnq5i": {
        "repo": "unsloth/gemma-3n-E4B-it-GGUF",
        "model": "gemma-3n-E4B-it-Q8_0.gguf"
    },
    "bafkreiaztifhss23cftya3bkorbsenzohol2oc3dvngo2srbbosko6gmme": {
        "repo": "LiquidAI/LFM2-1.2B-GGUF",
        "model": "LFM2-1.2B-Q8_0.gguf"
    },
    "bafkreidrdplo7mcfhrvocaa26yge6kmxmuwrexm5rffnzo5lbe6fkhjuvq": {
        "repo": "lmstudio-community/OpenReasoning-Nemotron-32B-GGUF",
        "model": "OpenReasoning-Nemotron-32B-Q8_0.gguf"
    },
    "bafkreih4xgr5t7yc3yooz6i6usgpwhggaobspmgut4rnu42gi6cv77o4em": {
        "repo": "mistralai/Devstral-Small-2507_gguf",
        "model": "Devstral-Small-2507-Q8_0.gguf"
    },
    "bafkreiaha3sjfmv4affmi5kbu6bnayenf2avwafp3cthhar3latmfi632u":{
        "repo": "NikolaSigmoid/FLUX.1-dev"
    },
    "bafkreidbaksrogxispjejczfj36vtf5uzsbjt7irspl6kckynz5u2ugzke": {
        "repo": "NikolaSigmoid/FLUX.1-dev-NSFW-Master"
    }
}