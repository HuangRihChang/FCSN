import torch
import numpy as np

from knapsack import knapsack, knapSack_improve

# def eval_metrics(y_pred, y_true):
#     overlap = np.sum(y_pred * y_true)
#     precision = overlap / (np.sum(y_pred) + 1e-8)
#     recall = overlap / (np.sum(y_true) + 1e-8)
#     if precision == 0 and recall == 0:
#         fscore = 0
#     else:
#         fscore = 2 * precision * recall / (precision + recall)
#     return [precision, recall, fscore]


# def select_keyshots(video_info, pred_score, selection_rate=0.15):
#     assert selection_rate <=1 and selection_rate >= 0

#     N = video_info['length'][()]
#     cps = video_info['change_points'][()]
#     weight = video_info['n_frame_per_seg'][()]
#     positions = video_info['picks'][()]
#     pred_score = np.array(pred_score.cpu().data)
#     pred_score = upsample(pred_score, N)
#     pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
#     selected = knapsack(pred_value, weight, int(selection_rate * N))
#     selected = selected[::-1]
#     key_labels = np.zeros(shape=(N, ))
#     for i in selected:
#         key_labels[cps[i][0]:cps[i][1]] = 1
#     return pred_score.tolist(), selected, key_labels.tolist()

# def upsample(down_arr, N):
#     up_arr = np.zeros(N)
#     ratio = N // 320
#     l = (N - ratio * 320) // 2
#     i = 0
#     while i < 320:
#         up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i]
#         l += ratio
#         i += 1
#     return up_arr

def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]  # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]
        # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]
        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())
        # Select the best shots using the knapsack implementation
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)
        selected = knapSack_improve(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))
        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1
        all_summaries.append(summary)
    return all_summaries

def evaluate_summary(predicted_summary, user_summary, eval_method="max"):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)