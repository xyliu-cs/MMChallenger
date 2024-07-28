from eval import read_json, write_json

def check_mcq_correctness(model_ans, label_ans):
    sanity_list = [label_ans, f'({label_ans})']
    for ans in sanity_list:
        if model_ans.startswith(ans):
            return True
    return False

def check_yn_correctness(model_ans, label_ans):
    return label_ans.lower() in model_ans.lower()


def find_mcq_errors(output_list: list, target="all") -> list:
    assert target in ["all", "action", "place"]
    ret_list = []
    if target == "all":
        for output_dict in output_list:
            id = output_dict["id"]
            category = output_dict["category"]
            mcq_ans = output_dict["MCQ_ans"]
            model_ans_lists = output_dict["mcq_model_ans"]
            local_dict = {"id": id, "category": category, "image": []}
            for iid, ans_list in enumerate(model_ans_lists):
                for ans in ans_list:
                    if not check_mcq_correctness(ans, mcq_ans):
                        local_dict["image"].append(iid+1)
                        break
            if local_dict["image"]:
                ret_list.append(local_dict)
    else:
        for output_dict in output_list:
            if output_dict["category"] == target:
                id = output_dict["id"]
                category = output_dict["category"]
                mcq_ans = output_dict["MCQ_ans"]
                model_ans_lists = output_dict["mcq_model_ans"]
                local_dict = {"id": id, "category": category, "image": []}
                for iid, ans_list in enumerate(model_ans_lists):
                    for ans in ans_list:
                        if not check_mcq_correctness(ans, mcq_ans):
                            local_dict["image"].append(iid+1)
                            break
                if local_dict["image"]:
                    ret_list.append(local_dict)
    return ret_list


def find_yn_errors(output_list: list, target="all") -> list:
    assert target in ["all", "action", "place"]
    ret_list = []
    if target == "all":
        for output_dict in output_list:
            id = output_dict["id"]
            category = output_dict["category"]
            model_ans_lists = output_dict["yn_model_ans"]
            local_dict = {"id": id, "category": category, "image": []}
            for iid, ans_list in enumerate(model_ans_lists):
                for ans in ans_list:
                    if not check_yn_correctness(ans, "Yes"):
                        local_dict["image"].append(iid+1)
                        break
            if local_dict["image"]:
                ret_list.append(local_dict)
    else:
        for output_dict in output_list:
            if output_dict["category"] == target:
                id = output_dict["id"]
                category = output_dict["category"]
                model_ans_lists = output_dict["yn_model_ans"]
                local_dict = {"id": id, "category": category, "image": []}
                for iid, ans_list in enumerate(model_ans_lists):
                    for ans in ans_list:
                        if not check_yn_correctness(ans, "Yes"):
                            local_dict["image"].append(iid+1)
                            break
                if local_dict["image"]:
                    ret_list.append(local_dict)
    return ret_list


if __name__ == "__main__":
    # output_path = "/120040051/test_resource/output_answers/llava15_7b_outputs_fmt.json"
    output_path = "/120040051/test_resource/output_answers/instructblip_7b_outputs.json"

    outputs = read_json(output_path)
    mcq_error_list = find_mcq_errors(outputs)
    yn_error_list = find_yn_errors(outputs)

    # print(mcq_error_list)
    print("mcq all: ", len(mcq_error_list))
    mcq_error_list_action = find_mcq_errors(outputs, target="action")
    print("mcq action: ", len(mcq_error_list_action))
    mcq_error_list_place = find_mcq_errors(outputs, target="place")
    print("mcq place: ", len(mcq_error_list_place))

    # print(yn_error_list)
    print("yn all: ", len(yn_error_list))
    yn_error_list_action = find_yn_errors(outputs, target="action")
    print("yn action: ", len(yn_error_list_action))
    yn_error_list_place = find_yn_errors(outputs, target="place")
    print("yn place: ", len(yn_error_list_place))

    print('\n')
    print(mcq_error_list)