submodule(sourcery_command_line_m) sourcery_command_line_s
  implicit none

contains

  module procedure flag_value

    integer argnum, arglen, flag_value_length
    character(len=:), allocatable :: arg


    flag_search: &
    do argnum = 1,command_argument_count()

      if (allocated(arg)) deallocate(arg)

      call get_command_argument(argnum, length=arglen)
      allocate(character(len=arglen) :: arg)
      call get_command_argument(argnum, arg)

      if (arg==flag) then
        call get_command_argument(argnum+1, length=flag_value_length)
        allocate(character(len=flag_value_length) :: flag_value)
        call get_command_argument(argnum+1, flag_value)
        exit flag_search
      end if
    end do flag_search

  end procedure

end submodule
