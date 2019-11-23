import argparse


def main(yolo_output_filepath, gt_filepath):
    preds = []
    with open(yolo_output_filepath) as opened_yolo_output_file:
        yolo_output = opened_yolo_output_file.readlines()[3:]
        for line_count in range(len(yolo_output)-1):
            if yolo_output[line_count].startswith('Enter'):
                if yolo_output[line_count + 1].startswith('Enter'):
                    preds.append(0)
                else:
                    preds.append(1)
                    while yolo_output[line_count + 1].startswith('tomato'):
                        line_count += 1

    with open(gt_filepath) as opened_gt_file:
        gts = opened_gt_file.readlines()
    gts = [1 if string == 'True\n' else 0 for string in gts]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pred, gt in zip(preds, gts):
        if gt == 1:
            if pred == gt:
                tp += 1
            else:
                fn += 1

        elif gt == 0:
            if pred == gt:
                tn += 1
            else:
                fp += 1

    error_rate = (fp+fn)/(tp+tn+fp+fn)

    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)
    print("error_rate", error_rate)

    return error_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-output-filepath", required=True, type=str, help="path to yolo output txtfile")
    parser.add_argument("--gt-filepath", required=True, type=str, help="path to gt txtfile")
    args = parser.parse_args()
    args = vars(args)
    main(**args)


