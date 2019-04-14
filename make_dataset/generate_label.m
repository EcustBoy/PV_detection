%%%����PV����Ķ�ֵ��ģ�ͷ��α߽���ǩ
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

%��ѭ�����ڴ��ļ��������ζ�ȡͼƬ����
fprintf('��ʼ������ǩ����...\n')
for id=78:img_num
    %���ж��Ƿ���ƥ���ǩ���ݣ������������˴�ѭ��
    fprintf('��ʼ������%d��\n',id)
    if id_seq(id,1)==0
        fprintf('δ���ҵ���%d��ԭʼͼƬ��Ӧ��ǩ����!����...\n',id)
        disp('----------------------------------')
        continue;
    end
    img_path=strcat(content(id).folder,strcat('\',content(id).name));
    fprintf('�Ӵ��̶�ȡ��%d��ͼƬ����\n',id)
    img=imread(img_path);
    %bbox_label��mask_label�����ڴ洢,�����ڿ��ӻ�
    bbox_label=img;%Bbox_label
    mask_label=zeros(size(img));%Mask_label
    central_num=id_seq(id,2)-id_seq(id,1);
    
    fprintf('��ʾ��%d��ͼƬ�������\n',id)
    figure(1);
    idc=num2str(id);
    title('ͼƬ���:idc');
    hold on;
    subplot(121)
    imshow(img);
    hold on;
    title(content(id).name);
    subplot(122)
    imshow(img);
    title('Bbox_label');
    hold on;

    %�ȱ��GT�����ĵ�
    fprintf('�궨��%d��ͼƬ��GT�����ĵ�...\n',id)
    for i=0:central_num
        subplot(122)
        bbox_id=id_seq(id,1)+i;
        scatter(round(image_central_point(bbox_id,2)),round(image_central_point(bbox_id,1)),'.b');
        bbox_label(round(image_central_point(bbox_id,2)),round(image_central_point(bbox_id,1)),:)=[0,0,255];
        hold on;
    end
    hold on;
    %���GT������㼰�߽�
    row=central_num+1;
    fprintf('�궨��%d��ͼƬ��GT�򶥵㼰�߽�...\n',id)
    for k=1:row 
    %�޳�NAN
        bbox_v_num=image_polygon_vertices(id_seq(id,1)+k-1,1);%ÿһ��bbox������
        %��Чx/y������������
        tempx=zeros(1,bbox_v_num);
        tempy=zeros(1,bbox_v_num);
        temp_idx=1;
        temp_idy=1;
        for p=2:bbox_v_num*2+1
            %������ż�ж�Ϊx��y����(ży/��x)
            if mod(p,2)==0
                tempy(1,temp_idy)=round(image_polygon_vertices(id_seq(id,1)+k-1,p));
                temp_idy=temp_idy+1;
            end
            if mod(p,2)==1
                tempx(1,temp_idx)=round(image_polygon_vertices(id_seq(id,1)+k-1,p));
                %���bboxһ�Զ���
                subplot(122)
                scatter(tempx(1,temp_idx),tempy(1,temp_idy-1),'.r');
                temp_idx=temp_idx+1;
                hold on;
            end
        end
        %���Ʊպ϶����GT��
        figure(1);
        subplot(122)
        plot([tempx,tempx(1,1)],[tempy,tempy(1,1)],'color','red');

       %ɸѡ��λ��GT���ڵ����ص�
       %����Ե�������ο��ҳ�xmax,xmin��ymax,ymin(��С������Χ)
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
        %���ϵõ�������[(xmin,xmax),(ymin,ymax)]
        x_ver=[tempx,round(image_polygon_vertices(id_seq(id,1)+k-1,2))];
        y_ver=[tempy,round(image_polygon_vertices(id_seq(id,1)+k-1,1))];
        %����[(xmin,xmax),(ymin,ymax)]���Ʊ߽��
        x_box=[xmin,xmax,xmax,xmin,xmin];
        y_box=[ymin,ymin,ymax,ymax,ymin];
        for x=xmin:xmax
            for y=ymin:ymax
                %�ٷ�ʵ�ʸ����ı�ǩ�����ܰ�����������
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
    fprintf('��%d��ͼƬ��ǩ�������\n',id)
    bbox_path=strcat('D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Bbox_label\',strcat('bbox_',content(id).name));
    %mask_path=strcat('D:\my_course\graduation_project\AerialImageSolarArray\3385804\label_image_data\Mask_label\',strcat('mask_',content(id).name));
    fprintf('�洢��%d��ͼƬ��Ӧ��ǩ...\n',id)
    imwrite(bbox_label,bbox_path);
    %imwrite(mask_label,mask_path);
    disp('----------------------------------')
end

