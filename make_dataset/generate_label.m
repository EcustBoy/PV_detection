%%%生成PV对象的二值掩模和方形边界框标签
base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804\raw_image_data';% the relative path of original images 
content=dir(strcat(base_path,'\*.tif'));
img_num=length(content);
% id_seq=zeros(img_num,2);
% data_num=length(image_name);
% flag=0;
% id1=0;
% id2=0;
% for i=1:img_num
%     name_=strsplit(content(i).name,'.');
%     name_=name_{1,1};
%     flag=0;
%     for j=1:data_num
%         if image_name(j)==string(name_)
%             if flag==0
%                 id1=j;
%                 flag=flag+1;
%             else
%                 id2=j;
%             end
%             id_seq(i,1)=id1;
%             id_seq(i,2)=id2;
%         end
%     end
% end

%1:lan lon
%2:lon lan
%lon<-->x;lan<-->y

%外循环用于从文件夹内批次读取图片数据
fprintf('开始制作标签数据...\n')
for id=78:img_num
    %先判断是否有匹配标签数据，若无则跳过此次循环
    fprintf('开始制作第%d张\n',id)
    if id_seq(id,1)==0
        fprintf('未能找到第%d张原始图片对应标签数据!跳过...\n',id)
        disp('----------------------------------')
        continue;
    end
    img_path=strcat(content(id).folder,strcat('\',content(id).name));
    fprintf('从磁盘读取第%d张图片数据\n',id)
    img=imread(img_path);
    %bbox_label与mask_label均用于存储,不用于可视化
    bbox_label=img;%Bbox_label
    mask_label=zeros(size(img));%Mask_label
    central_num=id_seq(id,2)-id_seq(id,1);
    
    fprintf('显示第%d张图片处理过程\n',id)
    figure(1);
    idc=num2str(id);
    title('图片序号:idc');
    hold on;
    subplot(121)
    imshow(img);
    hold on;
    title(content(id).name);
    subplot(122)
    imshow(img);
    title('Bbox_label');
    hold on;

    %先标出GT框中心点
    fprintf('标定第%d张图片各GT框中心点...\n',id)
    for i=0:central_num
        subplot(122)
        bbox_id=id_seq(id,1)+i;
        scatter(round(image_central_point(bbox_id,2)),round(image_central_point(bbox_id,1)),'.b');
        bbox_label(round(image_central_point(bbox_id,2)),round(image_central_point(bbox_id,1)),:)=[0,0,255];
        hold on;
    end
    hold on;
    %标出GT框各顶点及边界
    row=central_num+1;
    fprintf('标定第%d张图片各GT框顶点及边界...\n',id)
    for k=1:row 
    %剔除NAN
        bbox_v_num=image_polygon_vertices(id_seq(id,1)+k-1,1);%每一个bbox顶点数
        %有效x/y坐标数据向量
        tempx=zeros(1,bbox_v_num);
        tempy=zeros(1,bbox_v_num);
        temp_idx=1;
        temp_idy=1;
        for p=2:bbox_v_num*2+1
            %根据奇偶判断为x或y坐标(偶y/奇x)
            if mod(p,2)==0
                tempy(1,temp_idy)=round(image_polygon_vertices(id_seq(id,1)+k-1,p));
                temp_idy=temp_idy+1;
            end
            if mod(p,2)==1
                tempx(1,temp_idx)=round(image_polygon_vertices(id_seq(id,1)+k-1,p));
                %标出bbox一对顶点
                subplot(122)
                scatter(tempx(1,temp_idx),tempy(1,temp_idy-1),'.r');
                temp_idx=temp_idx+1;
                hold on;
            end
        end
        %标制闭合多边形GT框
        figure(1);
        subplot(122)
        plot([tempx,tempx(1,1)],[tempy,tempy(1,1)],'color','red');

       %筛选出位于GT框内的像素点
       %先针对单个多边形框找出xmax,xmin和ymax,ymin(缩小遍历范围)
        xmax=0;
        ymax=0;
        for j=1:bbox_v_num
            if j==1
                xmin=tempx(j);
                ymin=tempy(j);
            end
            if tempx(j)<xmin
                xmin=tempx(j);
            end
            if tempy(j)<ymin
                ymin=tempy(j);
            end
            if tempx(j)>xmax
                xmax=tempx(j);
            end
            if tempy(j)>ymax
                ymax=tempy(j);
            end
        end
        %以上得到遍历框[(xmin,xmax),(ymin,ymax)]
        x_ver=[tempx,round(image_polygon_vertices(id_seq(id,1)+k-1,2))];
        y_ver=[tempy,round(image_polygon_vertices(id_seq(id,1)+k-1,1))];
        %根据[(xmin,xmax),(ymin,ymax)]绘制边界框
        x_box=[xmin,xmax,xmax,xmin,xmin];
        y_box=[ymin,ymin,ymax,ymax,ymin];
        for x=xmin:xmax
            for y=ymin:ymax
                %官方实际给定的标签集可能包含负数坐标
                if x<=0
                    x=1;
                end
                if y<=0
                    y=1;
                end
            %subplot(132);
                [in,on]=inpolygon(x,y,x_box,y_box);
%                 if id==15
%                     fprintf('id=15 row=%d debug:\n',k);
%                     disp('----------')
%                     disp(x);disp(y);
%                 end
                %if in
                    %bbox_label(y,x,:)=[0,255,0];
                    %mask_label(y,x,:)=[255,255,255];
                %end
                if on
                    bbox_label(y,x,:)=[255,0,0];
                end   
            end
        end
    end
    fprintf('第%d张图片标签制作完毕\n',id)
    bbox_path=strcat('D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Bbox_label\',strcat('bbox_',content(id).name));
    %mask_path=strcat('D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\',strcat('mask_',content(id).name));
    fprintf('存储第%d张图片对应标签...\n',id)
    imwrite(bbox_label,bbox_path);
    %imwrite(mask_label,mask_path);
    disp('----------------------------------')
end

