% This MATLAB script can be used to reproduce the ERP classification accuracy for the dataset B
% Please download BBCItoolbox to 'MyToolboxDir'
% Please download dataset to 'EegMyDataDir'
% The authors would be grateful if published reports of research using this code
% (or a modified version, maintaining a significant portion of the original code) would cite the following article:
% Shin et al. "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset",
% Scientific data (2017), under review.
% NOTE: Figure may be different from that shown in Shin et al. (2017) because EOG-rejection is not performed.

clear all; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%% modify directory paths properly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MyToolboxDir = fullfile('/Users/BetaHouse/Stuffs/Mtech-IIITB/2019-Term-III/GEN_512_ML/Project/Analysis of EEG and NIRS data/bbci_public-master');
WorkingDir = fullfile('/Users/BetaHouse/Stuffs/Mtech-IIITB/2019-Term-III/GEN_512_ML/Project/Analysis of EEG and NIRS data');
% EegMyDataDir = fullfile('F:','scientific_data_publish','rawdata','EEG','without EOG');
EegMyDataDir = fullfile('/Users/BetaHouse/Stuffs/Mtech-IIITB/2019-Term-III/GEN_512_ML/Project/Analysis of EEG and NIRS data/EEG_01-26_MATLAB');
StatGNG = fullfile('/Users/BetaHouse/Stuffs/Mtech-IIITB/2019-Term-III/GEN_512_ML/Project/Analysis of EEG and NIRS data/behavior/Go-NoGo/summary');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',EegMyDataDir,'TmpDir','/tmp/','History',0);
cd(WorkingDir);

%% initial parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subdir_list = {'VP001-EEG','VP002-EEG','VP003-EEG','VP004-EEG','VP005-EEG','VP006-EEG','VP007-EEG','VP008-EEG','VP009-EEG','VP010-EEG','VP011-EEG','VP012-EEG','VP013-EEG','VP014-EEG','VP015-EEG','VP016-EEG','VP017-EEG','VP018-EEG','VP019-EEG','VP020-EEG','VP021-EEG','VP022-EEG','VP023-EEG','VP024-EEG','VP025-EEG','VP026-EEG'};
stimDef.eeg= {16, 32;'sym o','sym x'};

disp_ival = [-0.2 1] * 1000;
base_ival = [-0.1 0] * 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for vp = 1 : length(subdir_list)
    rng(vp); % for reproducibility
    disp([subdir_list{vp}, ' was started']);
     
    % correct epoch selection
    vpDir = fullfile(StatGNG, subdir_list{vp});
    cd(vpDir);
    load summary1; load summary2; load summary3;   
    
    summary = [summary1.result; summary2.result; summary3.result];
    summary = reshape(summary', 1, size(summary,1)*size(summary,2))';
    
    correctIdx = find(summary == 1);
    incorrectIdx = find(summary ~= 1);
    
    cd(WorkingDir);   
     
    % Load raw EEG data
    loadDir = fullfile(EegMyDataDir,subdir_list{vp});
    cd(loadDir);
    load cnt_dsr; load mrk_dsr; load mnt_dsr;
    cd(WorkingDir);

    % Marker definition
    mrk_dsr= mrk_defineClasses(mrk_dsr, stimDef.eeg);
    
    % Select EEG channels only (apart from EOG channels) for classification
    cnt_dsr = proc_selectChannels(cnt_dsr, 'not','*EOG'); % remove EOG channels (VEOG, HEOG)
    mnt_dsr = mnt_setElectrodePositions(cnt_dsr.clab);
                        
    % Segmentation
    epo = proc_segmentation(cnt_dsr, mrk_dsr, disp_ival);
    
    % Select epoch with correct answer
    epo = proc_selectEpochs(epo, 'not', incorrectIdx);
    disp([num2str(length(incorrectIdx)), ' epoch(s) was/were rejected due to incorrect answer']);

    % baseline correction and r^2
    epo= proc_baseline(epo, base_ival);
    epo_r= proc_rSquareSigned(epo);
    
    % Select discriminative time intervals
    dispclab = {'Fp1','AFF5h','AFz','F1','FC5','FC1','T7','C3','Cz','CP5','CP1','P7','P3','Pz','POz','O1','Fp2','AFF6h','F2','FC2','FC6','C4','T8','CP2','CP6','P4','P8','O2'};
    constraint= {{0, [0 200], dispclab, [200 400]},{0, [400 600], dispclab, [400 600]},{0, [600 800], dispclab, [600 800]}};
    ival_cfy{vp} = procutil_selectTimeIntervals(epo_r,'NIvals',5);
    
    % Feature extraction and classification
    fv = proc_jumpingMeans(epo, ival_cfy{vp});
    
%     computes size so that epoch(s) are not lost
    xsz= size(fv.x);
    fvsz= [prod(xsz(1:end-1)) xsz(end)];
    fv_x = reshape(fv.x, fvsz);
    
%     Transpose to have features in columns
    fv_x_t = transpose(fv_x);
    
    fv_y_t = transpose(fv.y);
    
    fv_x_y_t = [fv_x_t fv_y_t];
    
%     saves in the working directory
    csvwrite(strcat(subdir_list{vp}, '.csv'), fv_x_y_t);
    
    [loss(vp), losssem(vp)] = crossvalidation(fv, @train_RLDAshrink, 'SampleFcn', {@sample_KFold, [10 10]});
%     [loss(vp), losssem(vp)] = crossvalidation(fv, @train_RSLDAshrink, 'SampleFcn', {@sample_KFold, [10 10]});
    acc(vp) = 1-loss(vp);
    disp([subdir_list{vp}, ' = ', num2str(acc(vp))]);

end

% grand average
ave = mean(acc);
stdev = std(acc);
disp(['grand average classification accuracy = ',num2str(ave),'+-',num2str(stdev)]);
