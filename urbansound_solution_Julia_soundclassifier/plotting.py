import torchvision

def plot(inputs, labels, fold, batch_count, epoch, df, indices, outputs = None):
    picture_count  = 0

    for index, label in enumerate(labels):
        img = inputs[index]
        index_in_df = indices[index].item()
        source_path = df.loc[index_in_df,'relative_path'].split('/')[-2] + '_' + df.loc[index_in_df,'relative_path'].split('/')[-1]
        if outputs != None:
            prediction = outputs[index]
            torchvision.utils.save_image(img,'/Users/juliakisela/HKA/7.Semester/Seminararbeit/pictures/urban_sound_test/fold={}_batchcount={}_picturecount={}_label={}_prediction={}_sourcepath={}.png'
                                         .format(fold, batch_count, picture_count, label.item(), prediction.item(), source_path))
        else:
            torchvision.utils.save_image(img,'/Users/juliakisela/HKA/7.Semester/Seminararbeit/pictures/urban_sound_train/fold={}_epoch={}_batchcount={}_picturecount={}_label={}_sourcepath={}.png'
                                         .format(fold, epoch, batch_count, picture_count, label.item(),source_path))
        picture_count += 1
        if picture_count > 1:
            break