# from datasets import Dataset


class DataPreprocessStandard:

    # TODO: refactor to just use the lambda and have map in consumer.

    @staticmethod
    def synth_json(example, prompt=None):
        if not prompt:
            prompt = "Give me a sample json of basemats, dopands and dopants2basemats."
        return {
            "prompt": prompt,
            "completion": example["completion"],
        }
        # replace prompt with question for json output
        # dataset = dataset.map(
        #     lambda x:
        # )
        # return dataset

    # leave real extraction for time being... could do better with sys prompt though
    @staticmethod
    def synth_span(example, prefix=None):
        if not prefix:
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
    # save prompts here for now...
    synth_span_system_prompt = (
        "You are an expert writer."
        " I will give you a json and you will give me a sentence containing its elements."
        " In other words, if you were asked to extract the dopants from the sentence you created"
        " it should extract the json I give you."
        " Respond with the exact answer only, no explanations, no prefixes."
    )
    synth_json_system_prompt = (
        "You are a helpful assistant that generates json."
        " Respond with the exact answer only, no explanations, no prefixes."
    )
    
    synth_json_user_prompt = (
        "Give me a sample json of basemats, dopands and dopants2basemats."
    )
    
    extraction_system_prompt = (
        "Extract doping information from this sentence."
        " Respond with the exact answer only, no explanations, no prefixes."
    )
    # convert to Open AI style
    @staticmethod
    def synth_json(example, system_prompt=None, user_prompt=None):
        if not user_prompt:
            user_prompt = DataPreprocessOai.synth_json_user_prompt
        if not system_prompt:
            system_prompt = DataPreprocessOai.synth_json_system_prompt
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": example["completion"]},
            ]
        }

    @staticmethod
    def synth_span(example, system_prompt=None):
        if not system_prompt:
            system_prompt = DataPreprocessOai.synth_span_system_prompt
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{example['completion']}"},
                {
                    "role": "assistant",
                    "content": example["prompt"],
                },
            ]
        }

    @staticmethod
    def extraction(example, system_prompt=None):
        if not system_prompt:
            system_prompt = DataPreprocessOai.extraction_system_prompt
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": example["prompt"],
                },
                {"role": "assistant", "content": example["completion"]},
            ]
        }
