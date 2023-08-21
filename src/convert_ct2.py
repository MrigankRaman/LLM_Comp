from prompter import Prompter
from transformers import (
    LogitsProcessor,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
import os
import torch
from peft import PeftModel
import time
import ctranslate2
from transformers import AutoTokenizer
import subprocess
import shutil


def main(
    checkpoint_number=9000,
    merge_format=torch.float32,
    compilation_format="float32",
    to_test=False,
):
    compilation_formats = [
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    ]
    assert compilation_format in compilation_formats

    base_path = "meta-llama/Llama-2-13b-chat-hf"
    weights_path = "/home/mrigankraman/filestore-ai/mrigank/output_llama_13b_presidio/checkpoint-11123"
    tokenizer_path = "tokenizer"

    merged_weights_model_path = "merged_weights_llama2-13b-chat"
    ct2_target_path = f"/home/mrigankraman/filestore-ai/llama2-13b-chat-chkpt-{checkpoint_number}-merge_{str(merge_format).replace('.','_')}-ct2-{compilation_format}"

    # print("Merging Model")
    # make_merged_model(
    #     base_path=base_path,
    #     weights_path=weights_path,
    #     merge_format=merge_format,
    #     merged_weights_model_path=merged_weights_model_path,
    # )
    # print(f"Merged Model saved to {merged_weights_model_path}")

    # print("Moving tokenizer to merged model weights path.")
    # move_tokenizer_to_merged_model(
    #     tokenizer_path=tokenizer_path,
    #     merged_weights_model_path=merged_weights_model_path,
    # )

    print(f"Compiling Merged model to {compilation_format} in {ct2_target_path}")
    compile_merged_model(
        merged_weights_model_path=merged_weights_model_path,
        ct2_target_path=ct2_target_path,
        compilation_format=compilation_format,
    )

    if to_test:
        durations = []
        model_ct2 = ctranslate2.Generator(ct2_target_path, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        conversation = "Okay. How are you doing?  Well I am really tired.  Ahh yes your eyes are quite red as well.  How is your breathing?  Very bad i am short of breath often.  That is a shame to hear.  I hope you enjoy going to your sons baseball tournament.  Where is it?  Just in a local town called Johnston Village its about 10 minutes away.  My dad died at 55 of a heart attack.  My mom is in remission for breast cancer for 2 years. Do you smoke? yes i do. you should stop! so should you!  One pack a day for a year? That is really going to catch up to you"

        durations.extend(test_model(model_ct2, tokenizer, conversation))

        conversation = "Okay. Thank you for that. So, let, let me ask you a question. So, how's your energy level? Fine. Completely normal? I'm feeling great. I think I'm feeling better now than I even, I mean, I don't, what, I mean, I feel less stress after the tumor. I don't know if that, uh, I mean, after the surgery. I don't feel mad all the time. So, I don't know if that, uh, could have - You see, how much, uh, levothyroxine, thyroid pill, you're taking? Seventy - Seventy-five. Seventy-five, I think. Yeah. So, yeah. [INAUDIBLE] 75. So, we might as well do blood test now. It looks like, last time, your thyroid was good. Yeah. It was good. I think 75 is a good dose, but we'll check it again. What is it? What is the - Uh, thyroid. Thyroid. What's the normal, uh - Upper limit, I want it to be on upper limit of normal. All right. So, I want it to be, it's normal high 1.9. You were 1.4. So, I want it to be somewhere around 1.9 for you to have more energy. So, we'll check it - What am I now, 136 is my [INAUDIBLE]? 136. Yeah. All right. So, I mean, we are in the middle. Okay. 1.8 to 1.6, so, we're somewhere in the middle. So, I want it to be more. So, we'll do another blood test. We'll see if 75 or 88 is a good dose for you. Okay? So, we'll figure it out. Also, also, we will check your testosterone levels and see how you're doing, and, um - I feel even better than, uh, before the, the surgery. Okay. You still use spray? Yeah. I, uh, I have to use it. Okay. It doesn't, it gets better. I see. So, we have to monitor the question, if it gets better and when, but so, if you are, are you taking this once a week holiday of the spray? No. Because I am always in work or in school. I understand, but can you do it on Sunday? I work, or, I work or go to school the whole week. I never have, uh, like a day off for me. Okay. So, I want you to do, you know, whatever day off. I want you to take, because you take this in the morning, the spray, yes? So, I want you to, one day a week, make like a holiday, a little bit. So, I want you not to take it until you start peeing like crazy. Okay? Um - Can you do it? Yeah. So, you'll take it to your school. You'll not take it in the morning. You'll take it to your school, and when you notice that you're peeing, peeing a lot, you'll take it at school. Um-hum. But I want to have this period for your [INAUDIBLE] and balance. Okay? It's once a week we call holiday. Okay? [INAUDIBLE] holiday. So, it doesn't mean you don't take it this day, but you think that if you take it at 9:00, once a week, you don't take it at 9:00, and you, you'll just start taking it when you start to pee a lot. Okay? So, I want you to [INAUDIBLE], when you notice you pee one, two, like when it's like every hour or whatever, you'll start, you'll take it. Okay? Deal? Yeah. You bet. Again, so, you'll be able to do blood test for me? All right. When? Seventy-five. We'll do it, you're getting shot today. So, we'll do it in 10 days, blood work. Okay? So, doctor, how is the rest of my body? Is it, uh, is it getting a little better, or it's still the same? I mean, I mean, it looks, everything looks good. Can we see it again? Your blood work, what? Can we see it again? What? The - The blood work? Yeah. Sure. Let me show you, in a second. Let me just [INAUDIBLE], and this opens the screen that you need to read. Okay. And we'll figure it out. I think maybe the next time, we'll go to 88. So, I believe the 88 would be better, of levothyroxine. That would be a good dose for you, but we'll check on it. We'll check your chemistry. We will check your ACTH, check your lipids, check testosterone, PSA, prolactin. [INAUDIBLE]. So, all the blood test is here for you to do, and I'll see you back November. Oh. It will be thanksgiving. What about [DATE], before thanksgiving, Tuesday? All right. Good. And, uh, I have to give a prescription for the nurse to give you a shot. Good. Okay. So, you're all done. Let me give you your prescription, and I'll show you your blood work. Doctor? Yes? Can you give me like more pills per month? More pills for what? Like instead of giving me 30, 30 pills prescription, give me like 45 or 60. That way I don't have to - I can give you 90, but before I'll give you that - Um-hum. Uh, let's do it next time. I want to - All right. Figure out how much levothyroxine to give you. Okay? All right. If you need 75 or 88, which we'll know by next time. Okay. This is the prescription, and this is your blood work. So, we can go and look. So, this is your thyroid, which, so, it, uh, looks okay. Yeah. Testosterone was low last time, okay, when we did it. So, we'll check it now. So, it was on the low side, [INAUDIBLE]. Um, your, this was negative, the STD. What is that, the negative? Oh. The, the - It's all this STD stuff. Um, this is cholesterol. It's beautiful. Your chemistry looks good. So, there is nothing irregular. Everything looks normal here. What about the cortisol? Is it, does - We didn't check cortisol, but, uh, I think, if we checked it before it was undetectable in August. If you want to check, we can check it, but to check for cortisol, you have to, let's do it, [INAUDIBLE], but what I want from you, I want from you is at night is that you take a, let's say you want to do blood test on, on Wednesday. So, you take a prednisone on, on Tuesday, but on Wednesday, you take the prednisone after the blood work. Yeah. That's what, like I always do it like that. Yeah. Like you did it. Yeah. Yeah, so. Before the, when I take it, I wake up at 7:00, I take the, the pills at 7:00. After like the next day, I don't take it, and then, uh, I go do the blood work. You do blood work. Yeah. But what, even if I do that, if I come after school, I do the blood work, and then, um, I, I don't, but can I eat in the morning? Yeah. If I take the - But I want you to do like blood work in the morning, because I don't want you to be without the prednisone for more than a day - No. Because - For more than a day. That's not what I want. I want you to take it in the morning on Tuesday. On Wednesday, take it after your blood work, but I don't want your blood work to be at 6:00 PM. But the problem is that, uh, the, I go to school. No. You can't do it. It's dangerous. I don't want you to do it like that. Uh, I've done it before. No. I don't want you to do it. I want your, I want your cortisol in the morning. It's, it's useless to take your cortisol at 5:00 PM. I cannot calculate it. I cannot do anything about it. It should be done, it, it doesn't need to be fasting, but it needs to be at 8:00 or 9:00. At what time? 8:00 or 9:00, 7:30. You can come at 7:30 here, open, and you'll go back to your school, after that. Okay? This test should be done in the morning, or it's a waste of time. All right. So, you can get there a little bit late. If you need a note from us, we can give it to your school. Okay? All right. But otherwise, you do very well. Okay? Okay. So, go and get prescription, and get, go, go get, uh, shot. Okay? You're not open on Saturdays, right? Can I do the, the, the thing on, on Monday? You can do it on Monday. Yeah. Do the, because I don't have class on Monday. Do the cortisol though, because - I can do all the blood work on Monday. But, uh, what about testosterone? Because you're going to give me a shot today, right? Yeah. What if I don't take it? In two weeks. What if I - Can you do it on Monday in two weeks? No. No. What I'm saying like what if I don't take the testosterone today, and then, I come on Monday, and then, Monday, do the - No. I, I want your blood work in two weeks after your testosterone. Oh. So, all your blood work. So, it's not a problem. If you do your testosterone on Monday, you can come on Monday in two weeks and do blood work. Okay? Yeah. But if you do it today, you can also do it on Monday, in two weeks. No problem. Okay? Yeah. But it should be not [INAUDIBLE] the next day, that one, because I want to see in between average level. Because you know that the thing is that, uh, the problem is that they don't let me, they don't let me like, in the school, they don't let me, uh, like, they don't give, I cannot give them excuses. So, uh, do you just want to take the, the cortisol, do the cortisol? Do the cortisol whenever, like on Monday, I come at 7:30. Why, why would we do just the cortisol, if you do the same blood work at the same time? Because I cannot like, I have off on Monday. When? This Monday? Yeah. This Monday. So, I can do the cortisol. All right. You don't have Monday, every, every Monday. No. It's only this Monday. Yeah. So, I can come, uh, like - Let's do that. Okay. All right. We can do that. I come for, just for the cortisol. Yes. And then, next time that I come, I come, uh, I can come in the evening? Okay. Because I didn't understand. I'm thinking every Monday you have off. No. No. Okay. Good. No problem. Sounds good. Okay? All right. Take care. All the best. All right. Bye bye. Take care."

        durations.extend(test_model(model_ct2, tokenizer, conversation))

        print("Really long input")
        durations.extend(test_model(model_ct2, tokenizer, conversation * 3))

        print(durations)


def make_merged_model(base_path, weights_path, merge_format, merged_weights_model_path):
    if not os.path.exists(merged_weights_model_path):
        max_seq_len = 4096
        config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        config.update({"max_seq_len": max_seq_len})

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, config=config, device_map="cpu", torch_dtype=merge_format
        )

        lora_model = PeftModel.from_pretrained(
            base_model,
            weights_path,
            device_map="cpu",
            torch_dtype=merge_format,
        )
        merged_lora = lora_model.merge_and_unload()
        merged_lora.save_pretrained(merged_weights_model_path)


def move_tokenizer_to_merged_model(tokenizer_path, merged_weights_model_path):
    for fn in os.listdir(tokenizer_path):
        fp = os.path.join(tokenizer_path, fn)
        tp = os.path.join(merged_weights_model_path, fn)
        if not os.path.exists(tp):
            shutil.copyfile(fp, tp)


def compile_merged_model(
    merged_weights_model_path, ct2_target_path, compilation_format
):
    if not os.path.exists(ct2_target_path):
        command = f"ct2-transformers-converter --model {merged_weights_model_path} --quantization {compilation_format} --output_dir {ct2_target_path}"
        print(command)
        subprocess.run(
            command, text=True, check=True, shell=True, stdout=subprocess.PIPE
        )
    else:
        print(f"{ct2_target_path} already exists")


def test_model(model_ct2, tokenizer, conversation):
    # prompt_template = (
    #     "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\nConversation:\n{conversation}\n\n### Response:{prompt_seed}"
    #     ""
    # )
    prompter = Prompter(tokenizer=tokenizer, max_seq_len=4096)

    rest_instruction = "Write a Social History, Medications, Past Medical History, and Family Medical History section for the given patient-doctor conversation."
    rest_seed = "Social History:\n"

    ros_instruction = "Write a Review of Systems section for the given patient-doctor conversation. Make sure that the Review of Systems includes only the pertinent positive and negatives for the following subsections: Constitutional, Eye, Respiratory, Cardiovascular, Breast, Gastrointestinal, Genitourinary, Gynecologic, Heme or lymph, Endocrine, Immunologic, Musculoskeletal, Integumentary, Neurologic."
    ros_seed = "Review of systems:\n"

    sections = {
        "ros": {"instruction": ros_instruction, "seed": ros_seed},
        "other": {"instruction": rest_instruction, "seed": rest_seed},
    }

    generation_config = {
        "no_repeat_ngram_size": 10,
        "min_length": 4,
        "max_length": 512,
        "repetition_penalty": 1.01,
        "beam_size": 2,
        "sampling_topk": 2,
        "end_token": tokenizer.eos_token,
        'include_prompt_in_result': False
    }

    print("*"*50)
    durations = []
    for section_name, section_values in sections.items():
        prompt = prompter.generate_prompt(
            instruction=section_values["instruction"],
            conversation=conversation,
            prompt_seed=section_values["seed"],
            max_new_tokens=generation_config["max_length"],
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
        print(f"Input of {len(tokens):,} tokens")
        tstart = time.time()
        results = model_ct2.generate_batch(
            [tokens], **generation_config
        )
        duration = time.time() - tstart
        output = tokenizer.decode(results[0].sequences_ids[0])

        result = prompter.get_response(output)
        print("=" * 40)
        print(f"Section: {section_name}")
        print("___________")
        print(result)
        print(f"Took {duration:.2f} seconds")
        print("___________")
        durations.append(duration)
    return durations


if __name__ == "__main__":
    main(
        checkpoint_number=8600,
        compilation_format="bfloat16"
    )
