program concurrent_inferences
    !! This program demonstrates how to read a neural network from a JSON file
    !! and use the network to perform concurrent inferences.
    use inference_engine_m_, only : inference_engine_t, infer, parallel_infer
    use tensor_m, only: tensor_t
    use sourcery_file_m, only : file_t
    use sourcery_string_m, only : string_t
    use sourcery_command_line_m, only: command_line_t
    use iso_fortran_env, only : int64, real64
    use omp_lib
    implicit none
  
    type(string_t) network_file_name
    type(command_line_t) command_line
  
    !network_file_name = string_t(command_line%flag_value("--network"))
    network_file_name = "training_network.json"
    
    ! if (len(network_file_name%string())==0) then
    !   error stop new_line('a') // new_line('a') // &
    !     'Usage: fpm run --example concurrent-inferences --profile release --flag "-fopenmp" -- --network "<file-name>"'
    ! end if
    
    block 
      type(inference_engine_t) network, inference_engine
      type(tensor_t), allocatable :: inputs(:,:,:), outputs(:,:,:), outputs_elem_infer(:,:,:)
      real, allocatable :: input_components(:,:,:,:),outputs_elem_infer_values(:,:,:,:),outputs_values(:,:,:,:)
      real, parameter :: tolerance = 1.e6
      integer :: lat, lon, lev ! latitudes, longitudes, levels (elevations)
      integer i, j, k, e,l, dims_w(3),dims_b(2),dims_n(1)
      real n_operations_single_point, n_operations, error, byte_to_transfer, error_sum
      integer rep
      integer, parameter :: maxrep = 3, nPoints = 450*350*20
      real(real64) time_serial, time_parallel, time_exec
      integer, dimension(:), allocatable :: seed
      integer :: i, n, num_seed
      real :: rand

      ! Scopri il numero di elementi necessari per il seme
      call random_seed(size=num_seed)

      ! Alloca l'array per il seme e imposta i valori
      allocate(seed(num_seed))
      do i = 1, num_seed
          seed(i) = i + 56  ! Puoi impostare un valore specifico o usare un array
      end do

      ! Imposta il seme
      call random_seed(put=seed)

      print *, "Constructing a new inference_engine_t object from the file " 
      inference_engine = inference_engine_t()

      n_operations_single_point = 0
      do i = 1, ubound(inference_engine%nodes_,1)
        n_operations_single_point = n_operations_single_point + 2 * inference_engine%nodes_(i) * inference_engine%nodes_(i-1) + 2 * inference_engine%nodes_(i)
      end do
      dims_w = shape(inference_engine%weights_)
      dims_b = shape(inference_engine%biases_)
      dims_n = shape(inference_engine%nodes_)
      open(unit=1, file="results.csv", status='replace', action='write')
      write(1, '(A)') 'NPoints;'
      lev = 1
      lat = 1
      do lon=nPoints,nPoints+1,50000 
        n_operations = (lat * lev * lon) * n_operations_single_point / 10**9
        
        allocate(inputs(lon,lev,lat))
        allocate(outputs(lon,lev,lat))
        allocate(input_components(lon,lev,lat,inference_engine%num_inputs()))
        

        byte_to_transfer = (sizeof(input_components(1,1,1,1))*((lat*lon*lev)*(inference_engine%num_inputs()+inference_engine%num_outputs()) &
                  + dims_b(1)*dims_b(2) + dims_w(1)*dims_w(2)+dims_w(3) + 2*inference_engine%num_inputs() + 2*inference_engine%num_outputs())+ (sizeof(lon)*dims_n(1)))/10**6
        
        
        block 
          integer(int64) t_start, t_finish, clock_rate
          real(real64) t_exec
          time_serial = 0
          time_parallel = 0
          time_exec = 0
          error_sum = 0
          print *, lon
          do rep=1,maxrep
            call random_number(input_components)
            do concurrent(i=1:lon, j=1:lev, k=1:lat)
              inputs(i,j,k) = tensor_t(input_components(i,j,k,:))
            end do
            allocate(outputs_values(lon,lev,lat,inference_engine%num_outputs()))
            allocate(outputs_elem_infer_values(lon,lev,lat,inference_engine%num_outputs()))
            error = 0

            !print *,"Performing elemental inferences"
            call system_clock(t_start, clock_rate)
            outputs_elem_infer = inference_engine%infer(inputs)  ! implicit allocation of outputs array
            call system_clock(t_finish)
            time_serial = time_serial + real(t_finish - t_start, real64)/real(clock_rate, real64)
            !print *,"Elemental inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
            
            ! print *,"Performing loop-based inference"
            ! call system_clock(t_start)
            ! do k=1,lat
            !   do j=1,lev
            !     do i=1,lon
            !       outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
            !     end do 
            !   end do
            ! end do
            ! call system_clock(t_finish)
            ! print *,"Looping inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

            ! print *,"Performing concurrent inference"
            ! call system_clock(t_start)
            ! do concurrent(i=1:lon, j=1:lev, k=1:lat)
            !   outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))           
            ! end do
            ! call system_clock(t_finish)
            ! print *,"Concurrent inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
      
            ! print *,"Performing concurrent inference with a non-type-bound inference procedure"
            ! call system_clock(t_start)
            ! do concurrent(i=1:lon, j=1:lev, k=1:lat)
            !   outputs(i,j,k) = infer(inference_engine, inputs(i,j,k))           
            ! end do
            ! call system_clock(t_finish)
            ! print *,"Concurrent inference time with non-type-bound procedure: ", &
            ! real(t_finish - t_start, real64)/real(clock_rate, real64)

            ! print *,"Performing cuda inferences"
            ! call system_clock(t_start, clock_rate)
            ! call inference_engine%cuda_infer(inputs,outputs,t_exec)  ! implicit allocation of outputs array
            ! call system_clock(t_finish)
            ! time_exec = time_exec + t_exec
            ! time_parallel = time_parallel + real(t_finish - t_start, real64)/real(clock_rate, real64)
            ! print *,"Cuda inference time: ", &
            ! real(t_finish - t_start, real64)/real(clock_rate, real64) 
            ! print *, t_exec

            ! do concurrent(i=1:lon, j=1:lev, k=1:lat)
            !   outputs_elem_infer_values(i,j,k,:) = outputs_elem_infer(i,j,k)%values_ 
            !   outputs_values(i,j,k,:) = outputs(i,j,k)%values_ 
            ! end do

            ! do concurrent(i=1:lon, j=1:lev, k=1:lat, l=1:inference_engine%num_outputs())
            !   if (abs(outputs_elem_infer_values(i,j,k,l) - outputs_values(i,j,k,l)) > tolerance) then
            !     print *, outputs_elem_infer_values(i,j,k,l), outputs_values(i,j,k,l)
            !   end if  
            ! end do
    
            !print *,"Performing multithreading/offloading inferences"
            call system_clock(t_start, clock_rate)
            call inference_engine%parallel_infer(inputs,outputs,t_exec)  ! implicit allocation of outputs array
            call system_clock(t_finish)
            time_exec = time_exec + t_exec
            time_parallel = time_parallel + real(t_finish - t_start, real64)/real(clock_rate, real64)
            ! print *,"Multithreading/Offloading inference time: ", &
            ! real(t_finish - t_start, real64)/real(clock_rate, real64) 
            ! print *, t_exec

            do concurrent(i=1:lon, j=1:lev, k=1:lat)
              outputs_elem_infer_values(i,j,k,:) = outputs_elem_infer(i,j,k)%values_ 
              outputs_values(i,j,k,:) = outputs(i,j,k)%values_ 
            end do

            do i=1,lon
              do j=1,lev
                do k=1,lat
                  do e=1,inference_engine%num_outputs()
                    error = error + real((abs(outputs_values(i,j,k,e) - outputs_elem_infer_values(i,j,k,e))*tolerance))/real(abs(outputs_elem_infer_values(i,j,k,e)));
                  end do
                end do
              end do
            end do 
            error_sum = error_sum + real(error/(lon*lev*lat*inference_engine%num_outputs()))
            deallocate(outputs_elem_infer_values)
            deallocate(outputs_values)
          end do
        end block
        deallocate(inputs)
        deallocate(outputs)
        deallocate(input_components)
        
        time_parallel = time_parallel / maxrep
        time_serial = time_serial / maxrep
        time_exec = time_exec / maxrep
        error_sum = error_sum / maxrep
        
        write(1, '(I7,";",F12.8)') &!,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8,";",F12.8)') &
      lon, time_exec!time_serial, time_parallel, time_exec, real(n_operations)/time_serial, n_operations/time_exec, time_serial/time_parallel, time_serial/time_exec, error_sum, real(byte_to_transfer)/(time_parallel-time_exec)
      end do
      close(1)
    end block
  
  end program

  
  