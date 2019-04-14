%%%从5000*5000大图片分割50*50子图片
%读取原始图像mask标签
mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\raw_mask';% the relative path of original images 
%切割子图像的mask label存储路径
seg_mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\seg_mask\';
%读取原始整图像
raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\raw_images';
%切割原始图像存储路径
seg_raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\seg_images\';

mask_content=dir(strcat(mask_base_path,'\*.tif'));
raw_content=dir(strcat(raw_base_path,'\*.tif'));

img_num=length(mask_content);
p=100;
figure(3);
hold on;
%%读取Mask_label
for i=26:30
%     clear;
%     %读取原始图像mask标签
%     mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\raw_mask';% the relative path of original images 
%     %切割子图像的mask label存储路径
%     seg_mask_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\seg_mask\';
%     %读取原始整图像
%     raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\raw_images';
%     %切割原始图像存储路径
%     seg_raw_base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data\seg_images\';
%     mask_content=dir(strcat(mask_base_path,'\*.tif'));
%     raw_content=dir(strcat(raw_base_path,'\*.tif'));
%     img_num=length(mask_content);
%     p=100;
    
    fprintf('从磁盘读取原图像及其mask标签...\n');
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
    %割100*100小图片
    fprintf('开始扫描原mask图像...\n');
    for id_y=1:d
        for id_x=1:d
            sub_image=raw_mask(p*(id_x-1)+1:p*id_x,p*(id_y-1)+1:p*id_y);
            white_pixel_num=size(find(sub_image==255),1);
            %白色掩模像素数量占比
            white_per=white_pixel_num./10000;
            disp('--------');
            fprintf('扫描图片序号:%d\n',i);
            fprintf('the num of white pixels:');
            fprintf(num2str(white_pixel_num));
            fprintf('\n');
            s1=(id_y-1)*d+id_x;
            s2=d*d;
            per=s1./s2;
            x_min=(id_x-1)*p+1;x_max=id_x*p;
            y_min=(id_y-1)*p+1;y_max=id_y*p;
            %设定保留阈值
            if white_pixel_num>300    
                %此小块符合要求，保存
                %fprintf(strcat('完成度:',num2str(per),'%','\n'));
                fprintf('扫描完成度: %.2f%%.\n',per*100);
                fprintf(strcat('子图像:',num2str(id_x),'_',num2str(id_y),'匹配成功','\n'));
                fprintf('有效掩模像素索引及数目为:\n');
                white_pixel_num  
                find(sub_image==255)                 
                sub_index=strcat('_',num2str(id_x),'-',num2str(id_y));
                sub_image_name=strcat(sub_index,'_',num2str(white_per),'_',mask_content(i).name);
                segmentation_images_path_=strcat(segmentation_images_path,sub_image_name);
                %存储路径格式:seg_mask_base_path\i_id_x-id_y_name
                imwrite(sub_image,segmentation_images_path_);
                figure(3);
                subplot(121);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','green');
                hold on;
                %由当前有效方块坐标切割原图像并存储
                sub_raw_image=raw_image(p*(id_x-1)+1:p*id_x,p*(id_y-1)+1:p*id_y,1:3);
                %存储路径格式:seg_raw_base_path\i_id_x-id_y_name
                seg_raw_path=strcat(seg_raw_base_path,num2str(i),'_',num2str(id_x),'-',num2str(id_y),'_',num2str(white_per),'_',raw_images_name);
                imwrite(sub_raw_image,seg_raw_path);
                figure(3);
                subplot(122);
                plot([y_min,y_min,y_max,y_max,y_min],[x_min,x_max,x_max,x_min,x_min],'color','green');    
            else
                %fprintf(strcat('匹配失败,跳过:',num2str(per),'%','-',num2str(id_x),'-',num2str(id_y)));         
                %fprintf(strcat('完成度:',num2str(per),'%','\n'));
                fprintf('扫描完成度: %.2f%%.\n',per*100);
                fprintf(strcat('子图像',num2str(id_x),'_',num2str(id_y),'匹配失败，跳过','\n'));
                
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
    
            
    
        
    




