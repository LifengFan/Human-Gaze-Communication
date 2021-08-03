"""
Created on Oct 8, 2018

Author: Lifeng Fan

Description:

"""

from os import listdir
from os.path import isfile, join


def clean_annt(annot_path,NewAnnot_path,annot_files):
  for file_ind in range(len(annot_files)):

    file=annot_files[file_ind]
    vid=file.split('/')[-1][6:-4]

    if vid=='255' or vid=='291':
        continue

    print(vid)

    with open(file,'r') as file:
        lines=file.readlines()


    track_num=int(lines[-1].split(' ')[0])+1
    frame_num=len(lines)/track_num

    with open(NewAnnot_path+'NewAnt_'+vid+'.txt', "w") as to_write:

        # delete the final frame since there could be some errors in the final frame.
        for frame_ind in range(frame_num-1):

            targets=[]
            for track_ind in range(track_num):
                targets.append(lines[frame_ind+track_ind*frame_num])

            rec=[]

            for tar_ind in range(len(targets)):

                tgt=targets[tar_ind][:-1].split(' ')
                lost_flag=tgt[6]
                occ_flag=tgt[7]
                label=tgt[9]

                if int(lost_flag)==1 or int(occ_flag)==1:
                    continue
                elif label[1:7]=="Object":
                    for k in range(len(tgt)-1):
                        to_write.write(tgt[k])
                        to_write.write(" ")

                    to_write.write(tgt[-1][1:-1])
                    to_write.write("\n")

                else:

                      temp={"SmallAtt":"NA", "BigAtt":"NA", "Focus":"NA"}
                      temp["Prefix"]=tgt[0:9]
                      temp["Label"]=tgt[9][1:-1]

                      for k in range(10,len(tgt)):

                          tgt[k]=tgt[k][1:-1]

                          if tgt[k]=="single" or tgt[k]=="mutual" or tgt[k]=="avert" or tgt[k]=="refer" or tgt[k]=="follow" or tgt[k]=="share":
                              temp["SmallAtt"]=tgt[k]
                          elif tgt[k]=="SingleGaze" or tgt[k]=="GazeFollow" or tgt[k]=="AvertGaze" or tgt[k]=="MutualGaze" or tgt[k]=="JointAtt":
                              temp["BigAtt"]=tgt[k]
                          else:
                              temp["Focus"]=tgt[k]

                      rec.append(temp)

            for tar_ind1 in range(len(rec)-1):

                if rec[tar_ind1]["SmallAtt"]!="NA":
                    continue

                for tar_ind2 in range(tar_ind1+1,len(rec)):
                    if rec[tar_ind2]["SmallAtt"]!="NA":
                        continue

                    person1 = rec[tar_ind1]
                    person2 = rec[tar_ind2]

                    pid1=person1["Label"][-1]
                    pid2=person2["Label"][-1]

                    if person1["Focus"]=="P"+pid2:

                        if person2["Focus"]=="P"+pid1:
                            rec[tar_ind1]["SmallAtt"]="mutual"
                            rec[tar_ind2]["SmallAtt"]="mutual"

                        else:
                            pass

                    else:

                        if person2["Focus"]==person1["Focus"] and person1["Focus"]!="NA":

                            rec[tar_ind1]["SmallAtt"]="share"
                            rec[tar_ind2]["SmallAtt"]="share"

                        else:
                            pass

            for tar_ind in range(len(rec)):

                if rec[tar_ind]["SmallAtt"]=="NA":

                    rec[tar_ind]["SmallAtt"]="single"

            for tar_ind in range(len(rec)):

                to_write.write(' '.join(rec[tar_ind]["Prefix"]))
                to_write.write(" ")

                to_write.write(rec[tar_ind]["Label"])
                to_write.write(" ")

                to_write.write(rec[tar_ind]["BigAtt"])
                to_write.write(" ")

                to_write.write(rec[tar_ind]["SmallAtt"])
                to_write.write(" ")

                to_write.write(rec[tar_ind]["Focus"])
                to_write.write("\n")


if __name__=='__main__':

    annot_path = '/home/lfan/Dropbox/Projects/ICCV19/DATA/annotation/'
    NewAnnot_path = '/home/lfan/Dropbox/Projects/ICCV19/DATA/annotation_cleaned/'
    annot_files = [join(annot_path, f) for f in sorted(listdir(annot_path)) if isfile(join(annot_path, f))]

    clean_annt(annot_path,NewAnnot_path,annot_files)




