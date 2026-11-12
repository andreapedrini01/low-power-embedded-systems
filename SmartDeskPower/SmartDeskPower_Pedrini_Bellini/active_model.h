#ifndef ACTIVE_MODEL_H
#define ACTIVE_MODEL_H

#include "config.h"

#ifdef USE_FINETUNED_MODEL
  #include "model_finetuned.h"
#else
  #include "model.h"
#endif

#endif // ACTIVE_MODEL_H
