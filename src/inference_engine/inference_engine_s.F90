! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
submodule(inference_engine_m_) inference_engine_s
  use cudafor
  use cublas
  implicit none

  contains

  module procedure infer

    real(rkind), allocatable :: a(:,:)
    integer, parameter :: input_layer = 0
    integer k, l
    associate(w => self%weights_, b => self%biases_, n => self%nodes_, output_layer => ubound(self%nodes_,1))

      allocate(a(maxval(n), input_layer:output_layer))
      a(1:n(input_layer),input_layer) = (inputs%values_ - self%input_range_%minima_)/(self%input_range_%maxima_- self%input_range_%minima_)
      !a(1:n(input_layer),input_layer) = inputs%values()
      feed_forward: &
      do l = input_layer+1, output_layer
        associate(z => matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))
          a(1:n(l),l) = 1./(1.+exp(-z))
        end associate
      end do feed_forward
      outputs = tensor_t(self%output_range_%minima_ + a(1:n(output_layer), output_layer)*(self%output_range_%maxima_ - self%output_range_%minima_ ))
      !outputs = tensor_t(a(1:n(output_layer), output_layer))

    end associate

  end procedure


  attributes(global) subroutine cuda_function(input_components,output_components,w,b,n,min_in, max_in, min_out, max_out,lon,lev,lat)
    real(rkind), device :: w(:,:,:), b(:,:)
    integer, device :: n(:)
    integer, device :: lon,lev,lat
    real(rkind), shared :: w_s(32,32,4)
    real, device :: min_in(:), max_in(:), min_out(:), max_out(:)
    real, device :: input_components(:,:,:,:), output_components(:,:,:,:)
    integer :: input_layer = 1, output_layer=5, l, row, col, n_points
    !real, shared :: min_in_s(7), max_in_s(7), min_out_s(5), max_out_s(5)
    real(rkind), allocatable :: a(:,:)
    real(rkind) :: part_res
    integer(selected_int_kind(8)) :: tid_global
    integer nBlocks, curr_block,x,y,z,tid
    
    allocate(a(maxval(n), input_layer:output_layer))
    n_points = lon*lev*lat
    nBlocks = ceiling(real(n_points) / real(blockDim%x))
    curr_block = blockIdx%x -1
    
    tid = threadIdx%x -1
    do while(tid < 4096)
      z = tid / (32*32)
      y = mod(tid, 32*32) / 32
      x = mod(mod(tid, 32*32),32)
      w_s(x+1,y+1,z+1) = w(x+1,y+1,z+1)
      tid = tid + blockDim%x
    end do
    call syncthreads()
    tid = threadIdx%x
    do while (curr_block < nBlocks)
      tid_global = blockDim%x*curr_block + tid -1
      if (tid_global < n_points) then

        z = tid_global / (lon*lev)
        tid_global = mod(tid_global, lon*lev)
        y = tid_global / lon
        x = mod(tid_global,lon)

        a(1:n(input_layer),input_layer) = (input_components(x+1,y+1,z+1,:) - min_in(:))/(max_in(:) - min_in(:))
        !a(1:n(input_layer),input_layer) = input_components(x+1,y+1,z+1,:)
        feed_forward: &
        do l = input_layer+1, output_layer
        !Matrix multiplication
          do row=1,n(l)
            part_res = 0
            do col=1,n(l-1)
              part_res = part_res + w_s(row,col,l-1)*a(col,l-1)
            end do
            a(row,l) = 1./(1.+exp(-(part_res + b(row,l-1))))
          end do
    ! !       !a(1:n(l),l) = 1./(1.+exp(-(matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))))
        end do feed_forward
        !output_components(x+1,y+1,z+1,:) = a(1:n(output_layer), output_layer)
        
        output_components(x+1,y+1,z+1,:) = min_out(:) + a(1:n(output_layer), output_layer)*(max_out(:) - min_out(:))
      end if
    curr_block = curr_block + 66535
    end do
  end subroutine

  module procedure cuda_infer
    integer, parameter :: input_layer = 0 !questo mi sa che serve al device
    integer i,j,k, l, dims(3) !Dovrebbe stare solo qui
    integer :: dim_thread, dim_block
    !real(rkind), allocatable :: w(:,:,:), b(:,:)
    !integer, allocatable :: n(:)
    real(rkind), device, allocatable :: w_d(:,:,:), b_d(:,:)
    integer, device, allocatable :: n_d(:)
    integer, device :: lat,lon,lev
    integer :: output_layer !questo serve al device
    !real, allocatable, dimension(:) :: min_in, max_in, min_out, max_out
    real, device, allocatable, dimension(:) :: min_in_d, max_in_d, min_out_d, max_out_d
    real(rkind), allocatable :: input_components(:,:,:,:), output_components(:,:,:,:)
    real(rkind), device, allocatable :: input_components_d(:,:,:,:), output_components_d(:,:,:,:), a_d(:,:,:,:,:)
    integer(int64) t_start, t_finish, clock_rate

    w_d = self%weights_ 
    b_d = self%biases_
    n_d = self%nodes_
    output_layer = ubound(self%nodes_,1)
    min_in_d = self%input_range_%minima_
    max_in_d = self%input_range_%maxima_
    min_out_d = self%output_range_%minima_
    max_out_d = self%output_range_%maxima_
    dims = shape(inputs)
    lon = dims(1)
    lev = dims(2)
    lat = dims(3)
    allocate(input_components(dims(1),dims(2),dims(3),self%num_inputs()))
    allocate(input_components_d(dims(1),dims(2),dims(3),self%num_inputs()))
    allocate(output_components(dims(1),dims(2),dims(3),self%num_outputs()))
    allocate(output_components_d(dims(1),dims(2),dims(3),self%num_outputs()))

    do concurrent(i=1:dims(1), j=1:dims(2), k=1:dims(3))
      input_components(i,j,k,:) = inputs(i,j,k)%values_
    end do
    input_components_d = input_components

    dim_thread = dims(1)*dims(2)*dims(3)
    if (dim_thread > 512) then
      dim_thread = 512
    end if
    
    dim_block = ceiling(real(dims(1)*dims(2)*dims(3)) / real(dim_thread))

    if (dim_block > 65535) then
      dim_block = 65535
    end if
    print *, dim_block,dim_thread
    call system_clock(t_start, clock_rate)
    call cuda_function<<<dim_block,dim_thread>>>(input_components_d,output_components_d,w_d,b_d,n_d,min_in_d,max_in_d, min_out_d, max_out_d,lon,lev,lat)
    call system_clock(t_finish)
    t_exec = real(t_finish - t_start, real64)/real(clock_rate, real64)
    output_components = output_components_d

    allocate(outputs(dims(1),dims(2),dims(3)))
    do concurrent(i=1:dims(1), j=1:dims(2), k=1:dims(3))
      outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
    end do

    deallocate(input_components(dims(1),dims(2),dims(3),self%num_inputs()))
    deallocate(input_components_d(dims(1),dims(2),dims(3),self%num_inputs()))
    deallocate(output_components(dims(1),dims(2),dims(3),self%num_outputs()))
    deallocate(output_components_d(lon,lev,lat,self%num_outputs()))
   

  end procedure

  module procedure parallel_infer
    real(rkind), allocatable :: a(:,:)
    integer, parameter :: input_layer = 0
    integer i,j,k, l, dims(3), lat, lon, lev
    real(rkind), allocatable :: w(:,:,:), b(:,:)
    integer, allocatable :: n(:)
    integer output_layer
    real, allocatable, dimension(:) :: min_in, max_in, min_out, max_out
    real, allocatable :: input_components(:,:,:,:), output_components(:,:,:,:)
    integer(int64) t_start, t_finish, clock_rate

    dims = shape(inputs)
    w = self%weights_ 
    n = self%nodes_
    b = self%biases_
    output_layer = ubound(n,1)
    min_in = self%input_range_%minima_
    max_in = self%input_range_%maxima_
    min_out = self%output_range_%minima_
    max_out = self%output_range_%maxima_
    dims = shape(inputs)
    lon = dims(1)
    lev = dims(2)
    lat = dims(3)
    allocate(input_components(dims(1),dims(2),dims(3),self%num_inputs()))
    allocate(output_components(dims(1),dims(2),dims(3),self%num_outputs()))

    do concurrent(i=1:lon, j=1:lev, k=1:lat)
      input_components(i,j,k,:) = inputs(i,j,k)%values_
    end do

    allocate(a(maxval(n), input_layer:output_layer))
    allocate(outputs(dims(1),dims(2),dims(3)))

#ifndef __OFFLOADING
call system_clock(t_start, clock_rate)
  !$omp parallel do private(a) shared(input_components, output_components, n,w,b,output_layer, min_in,max_in,min_out,max_out)
#else
  !$omp target enter data map(to:input_components,n,w,b,min_in,max_in,min_out,max_out)
  !$acc data copyout(output_components) &
  !$acc present_or_copyin(input_components,n,w,b,min_in, max_in, min_out, max_out) 

  call system_clock(t_start, clock_rate)
  !$acc parallel loop collapse(3) private(a)
  !$omp target teams distribute parallel do firstprivate(a) collapse(3)

#endif
    do i=1,dims(1)
      do j=1,dims(2)
        do k=1,dims(3)
          a(1:n(input_layer),input_layer) = (input_components(i,j,k,:) - min_in)/(max_in- min_in)
          !a(1:n(input_layer),input_layer) = input_components(i,j,k,:)
          feed_forward: &
          do l = input_layer+1, output_layer
            a(1:n(l),l) = 1./(1.+exp(-(matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l))))
          end do feed_forward
          output_components(i,j,k,:) = min_out + a(1:n(output_layer), output_layer)*(max_out - min_out)     
          !output_components(i,j,k,:) = a(1:n(output_layer), output_layer)
        end do
      end do
    end do  
#ifndef __OFFLOADING    
  !$omp end parallel do
call system_clock(t_start, clock_rate)
#else
  !$omp end target teams distribute parallel do
  call system_clock(t_finish)
  !$acc end data
  !$omp target exit data map(from:output_components) map(release:input_components,n,w,b,min_in,max_in,min_out,max_out)

#endif
  do concurrent(i=1:lon, j=1:lev, k=1:lat)
    outputs(i,j,k) = tensor_t(output_components(i,j,k,:))
  end do
  t_exec = real(t_finish - t_start, real64)/real(clock_rate, real64)
end procedure


  pure subroutine inference_engine_consistency(self)

    type(inference_engine_t), intent(in) :: self

    integer, parameter :: input_layer=0
  end subroutine

  pure subroutine difference_consistency(self)

    type(difference_t), intent(in) :: self

    integer, parameter :: input_layer=0

  end subroutine

  module procedure construct_from_padded_arrays

    inference_engine%metadata_ = metadata
    inference_engine%weights_ = weights
    inference_engine%biases_ = biases
    inference_engine%nodes_ = nodes

    block
      integer i

      if (present(input_range)) then
        inference_engine%input_range_ = input_range
      else
        associate(num_inputs => nodes(lbound(nodes,1)))
          associate(default_minima => [(0., i=1,num_inputs)], default_maxima => [(1., i=1,num_inputs)])
            inference_engine%input_range_ = tensor_range_t("inputs", default_minima, default_maxima)
          end associate
        end associate
      end if

      if (present(output_range)) then
        inference_engine%output_range_ = output_range
      else
        associate(num_outputs => nodes(ubound(nodes,1)))
          associate(default_minima => [(0., i=1,num_outputs)], default_maxima => [(1., i=1,num_outputs)])
            inference_engine%output_range_ = tensor_range_t("outputs", default_minima, default_maxima)
          end associate
        end associate
      end if
    end block

    !call set_activation_strategy(inference_engine)

  end procedure construct_from_padded_arrays

  module procedure construct_from_json

    type(string_t), allocatable :: lines(:), metadata(:)
    type(tensor_range_t) input_range, output_range
    real, allocatable, dimension(:) :: minima_in, maxima_in, minima_out, maxima_out
    real(rkind), allocatable :: weights(:,:,:), biases(:,:)
    integer, allocatable :: nodes(:)
    real(rkind), allocatable :: hidden_weights(:,:,:)
    integer l, dims_3(3), dims_2(2), dims_1, i, j, n_in, n_out
    !lines = file_%lines()

    !l = 1

    !l = 2
    metadata = [string_t(""),string_t(""),string_t(""),string_t(""),string_t("false")]
    ! if (adjustl(lines(l)%string()) == '"metadata": {') then
    !   block
    !     character(len=:), allocatable :: justified_line
    !     do 
    !       l = l + 1
    !       justified_line = adjustl(lines(l)%string())
    !       if (justified_line == "},") exit
    !       metadata(findloc(key, trim(get_key_string(justified_line)), dim=1)) = get_key_value(justified_line)
    !     end do
    !     l = l + 1
    !   end block
    ! end if 

    !Get inference_engine data
    open(1, file = 'd_w.dat', status = 'old')
    open(2, file = 'd_b.dat', status = 'old')
    open(3, file = 'd_n.dat', status = 'old')
    open(4, file = 'd_dim.dat', status='old')
    open(5, file = 'd_range.dat', status='old')
    read(4,*) dims_3(:)
    read(4,*) dims_2(:)
    read(4,*) dims_1
    read(4,*) n_in, n_out

    allocate(weights(dims_3(1),dims_3(2),dims_3(3)))
    allocate(biases(dims_2(1),dims_2(2)))
    allocate(nodes(dims_1))
    allocate(input_range%minima_(n_in))
    allocate(input_range%maxima_(n_in))
    allocate(output_range%minima_(n_out))
    allocate(output_range%maxima_(n_out))

    read(5,*) input_range%minima_(:)
    read(5,*) input_range%maxima_(:)
    read(5,*) output_range%minima_(:)
    read(5,*) output_range%maxima_(:)
    
    do i=1, dims_3(1)
      do j=1, dims_3(2)
        read(1,*) weights(i,j,:)
      end do
    end do

    do i=1, dims_2(1)    
      read(2,*) biases(i,:)
    end do

    do i=1, dims_1    
      read(3,*) nodes(i)
    end do
    
    inference_engine = inference_engine_t(metadata, weights, biases, nodes, input_range, output_range)

    !call set_activation_strategy(inference_engine)

  contains

    pure function get_key_string(line) result(unquoted_key)
      character(len=*), intent(in) :: line
      character(len=:), allocatable :: unquoted_key
    
      associate(opening_key_quotes => index(line, '"'), separator => index(line, ':'))
        associate(closing_key_quotes => opening_key_quotes + index(line(opening_key_quotes+1:), '"'))
          unquoted_key = trim(line(opening_key_quotes+1:closing_key_quotes-1))
        end associate
      end associate
    end function

    function get_key_value(line) result(value_)
      character(len=*), intent(in) :: line
      type(string_t) value_

  #ifdef __INTEL_COMPILER
      character(len=:), allocatable :: text_after_colon
      integer :: opening_value_quotes, closing_value_quotes
      text_after_colon = line(index(line, ':')+1:)
      opening_value_quotes = index(text_after_colon, '"')
      closing_value_quotes = opening_value_quotes + index(text_after_colon(opening_value_quotes+1:), '"')
  #endif
  #ifndef __INTEL_COMPILER
      associate(text_after_colon => line(index(line, ':')+1:))
        associate(opening_value_quotes => index(text_after_colon, '"'))
          associate(closing_value_quotes => opening_value_quotes + index(text_after_colon(opening_value_quotes+1:), '"'))
  #endif
            if (any([opening_value_quotes, closing_value_quotes] == 0)) then
              value_ = string_t(trim(adjustl((text_after_colon))))
            else
              value_ = string_t(text_after_colon(opening_value_quotes+1:closing_value_quotes-1))
            end if
  #ifndef __INTEL_COMPILER
          end associate
        end associate
      end associate
  #endif
    end function

  end procedure construct_from_json


  module procedure subtract

    block
      integer l

      allocate(difference%weights_difference_, mold = self%weights_)
      allocate(difference%biases_difference_, mold = self%biases_)
      allocate(difference%nodes_difference_, mold = self%nodes_)

      difference%weights_difference_ = 0.
      difference%biases_difference_ = 0.
      difference%nodes_difference_ = 0.

      l = 0
      difference%nodes_difference_(l)  = self%nodes_(l) - rhs%nodes_(l)
    
      associate(n => self%nodes_)
        do concurrent(l = 1:ubound(n,1))
          difference%weights_difference_(1:n(l),1:n(l-1),l) = self%weights_(1:n(l),1:n(l-1),l) - rhs%weights_(1:n(l),1:n(l-1),l)
          difference%biases_difference_(1:n(l),l) = self%biases_(1:n(l),l) - rhs%biases_(1:n(l),l)
          difference%nodes_difference_(l) = self%nodes_(l) - rhs%nodes_(l)
        end do
      end associate

    end block

  end procedure

  module procedure norm 
    norm_of_self = maxval([abs(self%weights_difference_), abs(self%biases_difference_), real(abs(self%nodes_difference_))])
  end procedure

  module procedure num_outputs
    output_count = self%nodes_(ubound(self%nodes_,1))
  end procedure

  module procedure num_inputs
    input_count = self%nodes_(lbound(self%nodes_,1))
  end procedure

  module procedure nodes_per_layer
    node_count = self%nodes_
  end procedure

  module procedure skip
    use_skip_connections = self%metadata_(findloc(key, "usingSkipConnections", dim=1))%string() == "true"
  end procedure

end submodule inference_engine_s
