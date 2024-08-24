#ifdef __cplusplus
class SNPE
{
private:
    zdl::DlSystem::Runtime_t runtime;
    std::shared_ptr<zdl::SNPE::SNPE> snpe;
    std::string dlc_path;
public:
    SNPE(palantir_dnn_runtime);
    ~SNPE(void);
    int check_runtime();
    int init_network(const char *);
    int reload_network(const int, const int);
    int execute_byte(uint8_t*,float *, int);
    int execute_float(float *,float *, int);
};
#endif

//called from libvpx
#ifdef __cplusplus
extern "C" {
#endif
    void *snpe_alloc(palantir_dnn_runtime);
    void snpe_free(void *);
    int snpe_check_runtime(void *);
    int snpe_load_network(void *, const char *);
    int snpe_reload_network(void *, const int, const int);
    int snpe_execute_byte(void *, uint8_t*,float *, int);
    int snpe_execute_float(void *, float *,float *, int);
#ifdef __cplusplus
}
#endif


