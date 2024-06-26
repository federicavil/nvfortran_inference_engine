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
      real, allocatable :: input_components(:,:,:,:), outputs_values(:,:,:,:), outputs_elem_infer_values(:,:,:,:)
      real, parameter :: tolerance = 1.e-2
      integer, parameter :: lat=20, lon=350, lev=450 ! latitudes, longitudes, levels (elevations)
      !integer, parameter :: lat=9, lon=9, lev=9 ! latitudes, longitudes, levels (elevations)
      
      integer i, j, k, l
      
      
      print *, "Constructing a new inference_engine_t object from the file " 
      inference_engine = inference_engine_t()

      print *,"Defining an array of tensor_t input objects with random normalized components"
      allocate(inputs(lon,lev,lat))
      allocate(outputs(lon,lev,lat))
      allocate(input_components(lon,lev,lat,inference_engine%num_inputs()))
      allocate(outputs_elem_infer_values(lon,lev,lat,inference_engine%num_outputs()))
      allocate(outputs_values(lon,lev,lat,inference_engine%num_outputs()))
      call random_number(input_components)
  
      do concurrent(i=1:lon, j=1:lev, k=1:lat)
        inputs(i,j,k) = tensor_t(input_components(i,j,k,:))
      end do
      
      block 
        integer(int64) t_start, t_finish, clock_rate
        real(real64) t_exec
        print *,"Performing elemental inferences"
        call system_clock(t_start, clock_rate)
        outputs_elem_infer = inference_engine%infer(inputs)  ! implicit allocation of outputs array
        call system_clock(t_finish)
        print *,"Elemental inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
  
        print *,"Performing loop-based inference"
        call system_clock(t_start)
        do k=1,lat
          do j=1,lev
            do i=1,lon
              outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))
            end do 
          end do
        end do
        call system_clock(t_finish)
        print *,"Looping inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)

        print *,"Performing concurrent inference"
        call system_clock(t_start)
        do concurrent(i=1:lon, j=1:lev, k=1:lat)
          outputs(i,j,k) = inference_engine%infer(inputs(i,j,k))           
        end do
        call system_clock(t_finish)
        print *,"Concurrent inference time: ", real(t_finish - t_start, real64)/real(clock_rate, real64)
  
        print *,"Performing concurrent inference with a non-type-bound inference procedure"
        call system_clock(t_start)
        do concurrent(i=1:lon, j=1:lev, k=1:lat)
          outputs(i,j,k) = infer(inference_engine, inputs(i,j,k))           
        end do
        call system_clock(t_finish)
        print *,"Concurrent inference time with non-type-bound procedure: ", &
        real(t_finish - t_start, real64)/real(clock_rate, real64)

        print *,"Performing cuda inferences"
        call system_clock(t_start, clock_rate)
        call inference_engine%cuda_infer(inputs,outputs,t_exec)  ! implicit allocation of outputs array
        call system_clock(t_finish)
        print *,"Cuda inference time: ", &
        real(t_finish - t_start, real64)/real(clock_rate, real64) 
        print *, t_exec

        do concurrent(i=1:lon, j=1:lev, k=1:lat)
          outputs_elem_infer_values(i,j,k,:) = outputs_elem_infer(i,j,k)%values_ 
          outputs_values(i,j,k,:) = outputs(i,j,k)%values_ 
        end do

        do concurrent(i=1:lon, j=1:lev, k=1:lat, l=1:inference_engine%num_outputs())
          if (abs(outputs_elem_infer_values(i,j,k,l) - outputs_values(i,j,k,l)) > tolerance) then
            print *, outputs_elem_infer_values(i,j,k,l), outputs_values(i,j,k,l)
          end if  
        end do
 
        print *,"Performing multithreading/offloading inferences"
        call system_clock(t_start, clock_rate)
        call inference_engine%parallel_infer(inputs,outputs,t_exec)  ! implicit allocation of outputs array
        call system_clock(t_finish)
        print *,"Multithreading/Offloading inference time: ", &
        real(t_finish - t_start, real64)/real(clock_rate, real64) 
        print *, t_exec
        do concurrent(i=1:lon, j=1:lev, k=1:lat)
          outputs_elem_infer_values(i,j,k,:) = outputs_elem_infer(i,j,k)%values_ 
          outputs_values(i,j,k,:) = outputs(i,j,k)%values_ 
        end do

        do concurrent(i=1:lon, j=1:lev, k=1:lat, l=1:inference_engine%num_outputs())
          if (abs(outputs_elem_infer_values(i,j,k,l) - outputs_values(i,j,k,l)) > tolerance) then
            print *, outputs_elem_infer_values(i,j,k,l), outputs_values(i,j,k,l)
          end if  
        end do
        
      end block
    end block
  
  end program

  
  