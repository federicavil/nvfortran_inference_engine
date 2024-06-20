submodule(tensor_m) tensor_s
  implicit none

contains

    module procedure construct_from_components
      tensor%values_ = values
    end procedure

    module procedure values
      tensor_values = self%values_
    end procedure

    module procedure num_components
      n = size(self%values_)
    end procedure

end submodule tensor_s