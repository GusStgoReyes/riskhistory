# Sigmoid function for computing entropy from subjective value
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Compute entropy from subjective value
entropy_from_sv <- function(sv) {
  p_accept <- sigmoid(sv)
  entropy <- - (p_accept * log2(p_accept) + (1 - p_accept) * log2(1 - p_accept))
  return(entropy)
}