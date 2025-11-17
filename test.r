# install.packages("tidyverse")
library(tidyverse)

# Create a data frame
df <- tibble(
  gyökér = c(13123141324, 341234, 6345636251),
  szignál = c(14234123523452, 123412341234, 1423412341234)
)

# Print the data frame
print(df)