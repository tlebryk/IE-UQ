# from datasets import Dataset


class DataPreprocessStandard:

    # TODO: refactor to just use the lambda and have map in consumer.

    @staticmethod
    def synth_json(example):
        question = "Give me a sample json of basemats, dopands and dopants2basemats."
        return {
            "prompt": question,
            "completion": example["completion"],
        }
        # replace prompt with question for json output
        # dataset = dataset.map(
        #     lambda x:
        # )
        # return dataset

    # leave real extraction for time being... could do better with sys prompt though
    @staticmethod
    def synth_span(example):
        prefix = (
            "Based on this json, give me a passage containing it's elements."
            " In other words, if you were asked to extract from the passage you created"
            " it would extract the json below."
            "\n\n###\n"
        )
        return {
            "prompt": prefix + example["completion"],
            "completion": example["prompt"],
        }

    @staticmethod
    def extraction(example):
        return {
            "prompt": example["prompt"],
            "completion": example["completion"],
        }


class DataPreprocessOai:
    # convert to Open AI style
    @staticmethod
    def synth_json(example):
        question = "Give me a sample json of basemats, dopands and dopants2basemats."
        return {
            "messages": [
                {"role": "system", "content": question},
                # {"role": "user", "content": example["prompt"]
                {"role": "assistant", "content": example["completion"]},
            ]
        }

    @staticmethod
    def synth_span(example):
        prefix = (
            "Based on this json, give me a sentence containing its elements."
            " In other words, if you were asked to extract the dopants from the sentence you created"
            " it should extract the json below."
        )
        return {
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": example["completion"]},
                {
                    "role": "assistant",
                    "content": example["prompt"],
                },
            ]
        }

    @staticmethod
    def extraction(example):
        prefix = "Extract doping information from this sentence."
        return {
            "messages": [
                {"role": "system", "content": prefix},
                {
                    "role": "user",
                    "content": example["prompt"],
                },
                {"role": "assistant", "content": example["completion"]},
            ]
        }
