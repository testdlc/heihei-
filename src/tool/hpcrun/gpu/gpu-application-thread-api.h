//******************************************************************************
// forward type declarations
//******************************************************************************

typedef struct cct_node_t cct_node_t;



//******************************************************************************
// interface operations
//******************************************************************************

void
gpu_application_thread_process_activities
(
 void
);


cct_node_t *
gpu_application_thread_correlation_callback
(
 uint64_t correlation_id
);