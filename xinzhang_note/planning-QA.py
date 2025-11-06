# TODO constants.py 中的常量表如何使用？ 定义 waypoint_ego? 


def add_special_token(special_token_list, tokenizer, model):
    # 给新的token添加索引并用大模型的embeding的平均值来初始化token的embeding
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens = True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg