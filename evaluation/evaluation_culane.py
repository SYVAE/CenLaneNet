import os

def eval_lane(data_root, work_dir,ifvalidation=False):
    if ifvalidation:
        res = call_culane_val(data_root, work_dir)
    else:
        res=call_culane_eval(data_root, work_dir)
    TP, FP, FN = 0, 0, 0
    for k, v in res.items():
        val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
        val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
        TP += val_tp
        FP += val_fp
        FN += val_fn
        print(k, val)
    P = TP * 1.0 / (TP + FP)
    R = TP * 1.0 / (TP + FN)
    F = 2 * P * R / (P + R)
    print(F)
    return F

def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res

def call_culane_eval(data_dir, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir =output_path+ '/'
    w_lane = 30
    iou = 0.5  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir, 'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir, 'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir, 'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir, 'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir, 'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir, 'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir, 'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir, 'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out0 = os.path.join(output_path, 'out0_normal.txt')
    out1 = os.path.join(output_path, 'out1_crowd.txt')
    out2 = os.path.join(output_path, 'out2_hlight.txt')
    out3 = os.path.join(output_path, 'out3_shadow.txt')
    out4 = os.path.join(output_path, 'out4_noline.txt')
    out5 = os.path.join(output_path, 'out5_arrow.txt')
    out6 = os.path.join(output_path, 'out6_curve.txt')
    out7 = os.path.join(output_path, 'out7_cross.txt')
    out8 = os.path.join(output_path, 'out8_night.txt')

    eval_cmd = './evaluation/evaluate'
    print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list1, w_lane, iou, im_w, im_h, frame, out1))
    print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list2, w_lane, iou, im_w, im_h, frame, out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list3, w_lane, iou, im_w, im_h, frame, out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list4, w_lane, iou, im_w, im_h, frame, out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list5, w_lane, iou, im_w, im_h, frame, out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list6, w_lane, iou, im_w, im_h, frame, out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list7, w_lane, iou, im_w, im_h, frame, out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list8, w_lane, iou, im_w, im_h, frame, out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd'] = read_helper(out1)
    res_all['res_night'] = read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow'] = read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve'] = read_helper(out6)
    res_all['res_cross'] = read_helper(out7)
    return res_all


def call_culane_val(data_dir, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir =output_path+ '/'
    w_lane = 30
    iou = 0.5  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/val.txt')

    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out0 = os.path.join(output_path, 'output_val.txt')
    eval_cmd = './evaluation/evaluate'
    print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -v %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0,'1'))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -v %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0,'1'))
    res_all = {}
    res_all['res_val'] = read_helper(out0)
    return res_all