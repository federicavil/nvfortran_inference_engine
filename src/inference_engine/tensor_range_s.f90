! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
submodule(tensor_range_m) tensor_range_s
  !use sourcery_formats_m, only : separated_values
  use kind_parameters_m, only : rkind
  implicit none
  
contains

  module procedure from_components
    tensor_range%layer_ = layer
    tensor_range%minima_ = minima
    tensor_range%maxima_ = maxima 
  end procedure

  module procedure from_json
    logical tensor_range_key_found
    integer l 

    tensor_range_key_found = .false.

    do l=1,size(lines)
      if (lines(l)%get_json_key() == "tensor_range") then
        tensor_range_key_found = .true.
        tensor_range%layer_  = lines(l+1)%get_json_value(key=string_t("layer"), mold=string_t(""))
        tensor_range%minima_ = lines(l+2)%get_json_value(key=string_t("minima"), mold=[0.])
        tensor_range%maxima_ = lines(l+3)%get_json_value(key=string_t("maxima"), mold=[0.])
        return
      end if
    end do 

  end procedure

  module procedure equals
    real, parameter :: tolerance = 1.E-08

    lhs_equals_rhs = &
      lhs%layer_ == rhs%layer_ .and. &
      all(abs(lhs%minima_ - rhs%minima_) <= tolerance).and. &
      all(abs(lhs%maxima_ - rhs%maxima_) <= tolerance)
  end procedure 

  ! module procedure to_json
  !   integer, parameter :: characters_per_value=17
  !   character(len=*), parameter :: indent = repeat(" ",ncopies=4)
  !   character(len=:), allocatable :: csv_format, minima_string, maxima_string

  !   csv_format = separated_values(separator=",", mold=[real(rkind)::])
  !   allocate(character(len=size(self%minima_)*(characters_per_value+1)-1)::minima_string)
  !   allocate(character(len=size(self%maxima_)*(characters_per_value+1)-1)::maxima_string)
  !   write(minima_string, fmt = csv_format) self%minima_
  !   write(maxima_string, fmt = csv_format) self%maxima_
  !   lines = [ &
  !     string_t(indent // '"tensor_range": {'), &
  !     string_t(indent // '  "layer": "' // trim(adjustl(self%layer_)) // '",'), &
  !     string_t(indent // '  "minima": [' // trim(adjustl(minima_string)) // '],'), & 
  !     string_t(indent // '  "maxima": [' // trim(adjustl(maxima_string)) // ']'), &
  !     string_t(indent // '}') &
  !   ]
  ! end procedure

  module procedure map_to_training_range
    associate(tensor_values => tensor%values())
      associate(normalized_values => (tensor_values - self%minima_)/(self%maxima_ - self%minima_))
        normalized_tensor = tensor_t(normalized_values)
      end associate
    end associate
  end procedure

  module procedure map_from_training_range
    associate(tensor_values => tensor%values())
      associate(unnormalized_values => self%minima_ + tensor_values*(self%maxima_ - self%minima_))
        unnormalized_tensor = tensor_t(unnormalized_values)
      end associate
    end associate
  end procedure

  module procedure in_range
    is_in_range = all(tensor%values() >= self%minima_) .and. all(tensor%values() <= self%maxima_)
  end procedure

end submodule tensor_range_s
