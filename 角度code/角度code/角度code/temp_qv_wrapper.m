
            function [faip, thetap] = temp_qv_wrapper()
                % 运行qv.m并捕获结果
                run('qv.m');
                % 假设qv.m创建了faip和thetap变量
            end
            