#ifndef ACTIVE_MODEL_H
#define ACTIVE_MODEL_H

#include "config.h"

// Include calibrated model if USE_FINETUNED_MODEL is defined, otherwise use base model.
#ifdef USE_FINETUNED_MODEL
  #include "model_finetuned.h"
#else
  #include "model.h"
#endif

#endif // ACTIVE_MODEL_H
