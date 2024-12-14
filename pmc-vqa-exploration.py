from datasets import load_dataset

data_files = {
'train': 'https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/train.csv',
'test': 'https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/test.csv',
'test_clean': 'https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/test_clean.csv',
}

dataset = load_dataset('csv', data_files=data_files, cache_dir="data/PMC-VQA")

data_test = dataset["test"]

for example in iter(data_test):
    image_name = example.get("Figure_path")
    question = example.get("Question")
    answer = example.get("Answer")
    choice_A = example.get("Choice A")
    choice_B = example.get("Choice B")
    choice_C = example.get("Choice C")
    choice_D = example.get("Choice D")
    answer_label = example.get("Answer_label")

    print(f"Image name: {image_name}")
    print(f"{question}")
    print(choice_A)
    print(choice_B)
    print(choice_C)
    print(choice_D)
    print(f"Answer: {answer_label}: {answer}")

    print("\n\n")
