%%%��5000*5000��ͼƬ�ָ�50*50��ͼƬ
%��ȡԭʼͼ��mask��ǩ
mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\raw_mask';% the relative path of original images 
%�и���ͼ���mask label�洢·��
seg_mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\seg_mask\';
%��ȡԭʼ��ͼ��
raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\raw_images';
%�и�ԭʼͼ��洢·��
seg_raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\seg_images\';

mask_content=dir(strcat(mask_base_path,'\*.tif'));
raw_content=dir(strcat(raw_base_path,'\*.tif'));

img_num=length(mask_content);
p=100;
figure(3);
hold on;
%%��ȡMask_label
for i=26:30
%     clear;
%     %��ȡԭʼͼ��mask��ǩ
%     mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\raw_mask';% the relative path of original images 
%     %�и���ͼ���mask label�洢·��
%     seg_mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\seg_mask\';
%     %��ȡԭʼ��ͼ��
%     raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\raw_images';
%     %�и�ԭʼͼ��洢·��
%     seg_raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\seg_images\';
%     mask_content=dir(strcat(mask_base_path,'\*.tif'));
%     raw_content=dir(strcat(raw_base_path,'\*.tif'));
%     img_num=length(mask_content);
%     p=100;
    
    fprintf('�Ӵ��̶�ȡԭͼ����mask��ǩ...\n');
    path=strcat(mask_content(i).folder,strcat('\',mask_content(i).name));
    segmentation_images_path=strcat(seg_mask_base_path,num2str(i));
    raw_mask=imread(path); 
    figure(3);
    title(strcat(num2str(i),'/',num2str(img_num)));
    hold on;
    subplot(121);
    imshow(raw_mask);
    hold on;
    c=strsplit(mask_content(i).name,'_');
    raw_images_name=c{1,2};
    path_=strcat(raw_content(i).folder,strcat('\',raw_images_name));
    raw_image=imread(path_);
    subplot(122);
    imshow(raw_image);
    hold on;
    
    shape=size(raw_mask);
    h=shape(1,1);
    w=shape(1,2);
    d=h./p;
    %��100*100СͼƬ
    fprintf('��ʼɨ��ԭmaskͼ��...\n');
    for id_y=1:d
        for id_x=1:d
            sub_image=raw_mask(p*(id_x-1)+1:p*id_x,p*(id_y-1)+1:p*id_y);
            white_pixel_num=size(find(sub_image==255),1);
            %��ɫ��ģ��������ռ��
            white_per=white_pixel_num./10000;
            disp('--------');
            fprintf('ɨ��ͼƬ���:%d\n',i);
            fprintf('the num of white pixels:');
            fprintf(num2str(white_pixel_num));
            fprintf('\n');
            s1=(id_y-1)*d+id_x;
            s2=d*d;
            per=s1./s2;
            x_min=(id_x-1)*p+1;x_max=id_x*p;
            y_min=(id_y-1)*p+1;y_max=id_y*p;
            %�趨������ֵ
            if white_pixel_num>300    
                %��С�����Ҫ�󣬱���
                %fprintf(strcat('��ɶ�:',num2str(per),'%','\n'));
                fprintf('ɨ����ɶ�: %.2f%%.\n',per*100);
                fprintf(strcat('��ͼ��:',num2str(id_x),'_',num2str(id_y),'ƥ��ɹ�','\n'));
                fprintf('��Ч��ģ������������ĿΪ:\n');
                white_pixel_num  
                find(sub_image==255)                 
                sub_index=strcat('_',num2str(id_x),'-',num2str(id_y));
                sub_image_name=strcat(sub_index,'_',num2str(white_per),'_',mask_content(i).name);
                segmentation_images_path_=strcat(segmentation_images_path,sub_image_name);
                %�洢·����ʽ:seg_mask_base_path\i_id_x-id_y_name
                imwrite(sub_image,segmentation_images_path_);
                figure(3);
                subplot(121);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','green');
                hold on;
                %�ɵ�ǰ��Ч���������и�ԭͼ�񲢴洢
                sub_raw_image=raw_image(p*(id_x-1)+1:p*id_x,p*(id_y-1)+1:p*id_y,1:3);
                %�洢·����ʽ:seg_raw_base_path\i_id_x-id_y_name
                seg_raw_path=strcat(seg_raw_base_path,num2str(i),'_',num2str(id_x),'-',num2str(id_y),'_',num2str(white_per),'_',raw_images_name);
                imwrite(sub_raw_image,seg_raw_path);
                figure(3);
                subplot(122);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','green');    
            else
                %fprintf(strcat('ƥ��ʧ��,����:',num2str(per),'%','-',num2str(id_x),'-',num2str(id_y)));         
                %fprintf(strcat('��ɶ�:',num2str(per),'%','\n'));
                fprintf('ɨ����ɶ�: %.2f%%.\n',per*100);
                fprintf(strcat('��ͼ��',num2str(id_x),'_',num2str(id_y),'ƥ��ʧ�ܣ�����','\n'));
                
                figure(3);
                subplot(121);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','red');
                hold on;
                figure(3);
                subplot(122);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','red');
                hold on;
            end
            
        end
    end
end
    
            
    
        
    




