import random

def disturb_rationale(file_list, batch_size, ratio=0.2):
    assert len(file_list) % batch_size == 0, 'wrong batch size'
    element_cnt = len(file_list) // batch_size
    shuffle_cnt = int(batch_size * ratio)

    # samples the
    selected_indices = []
    for i in range(element_cnt):
        selected_indices.append(tuple(random.sample(range(i*batch_size, (i+1)*batch_size), shuffle_cnt)))
    # shuffle the indices
    shuffled_indices = selected_indices.copy()
    iter_cnt = 0
    while any(shuffled_indices[i] == selected_indices[i] for i in range(element_cnt)) and iter_cnt < 500:
        random.shuffle(selected_indices)
        iter_cnt += 1
    if iter_cnt == 500:
        print("Warning: max iteration reached. The list may not be perfectly shuffled.")
    # conduct the swapping
    disturbed_list = file_list.copy()
    for i, new_idx in enumerate(shuffled_indices):
        ori_idx = selected_indices[i]
        for j in range(len(ori_idx)):
            disturbed_list[ori_idx[j]] = file_list[new_idx[j]]

    return disturbed_list


if __name__=='__main__':
    list_file = [{'id': i, 'data': f'data_{i}'} for i in range(800)]
    lst = disturb_rationale(list_file, 10)
    print(f'disturbed count: {sum([lst[i]['id'] != i for i in range(len(lst))])}')