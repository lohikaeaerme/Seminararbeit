import torchvision

def plot(inputs, labels, batch_count, epoch, df, indices, outputs = None):
    picture_count  = 0

    for index, label in enumerate(labels):
        img = inputs[index]
        index_in_df = indices[index].item()
        source_path = df.loc[index_in_df,'relative_path'].split('/')[-2] + '_' + df.loc[index_in_df,'relative_path'].split('/')[-1]
        if outputs != None:
            prediction = outputs[index]
            torchvision.utils.save_image(img,'/Users/juliakisela/HKA/7.Semester/Seminararbeit/pictures/audioMNIST_test/batchcount={}_picturecount={}_label={}_prediction={}_sourcepath={}.png'
                                         .format(batch_count, picture_count, label.item(), prediction.item(), source_path))
        else:
            torchvision.utils.save_image(img,'/Users/juliakisela/HKA/7.Semester/Seminararbeit/pictures/audioMNIST_train/epoch={}_batchcount={}_picturecount={}_label={}_sourcepath={}.png'
                                         .format(epoch, batch_count, picture_count, label.item(),source_path))
        picture_count += 1
        if picture_count > 1:
            break