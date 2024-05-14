import habitat_sim
import os
from loguru import logger

logger.info("Inside add_object.py: Current working directory: {}".format(os.getcwd()))

def recompute_navmesh(sim):
    # recompute the NavMesh with STATIC objects
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.include_static_objects = True
    navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    assert navmesh_success, "Failed to build static navmesh"

# Add objects to the scene based on template_id
def add_object(env, position, scale, template_id, navmesh_recompute=True):
    _sim = env.habitat_env.sim
    obj_template_mgr = _sim.get_object_template_manager()
    rigid_obj_mgr = _sim.get_rigid_object_manager()
    obj_template = obj_template_mgr.get_template_by_id(template_id)
    obj_template.scale *= scale
    # Define a new template for the new scale
    new_temp = obj_template.handle+'_new'
    new_tempid = obj_template_mgr.register_template(obj_template, new_temp)
    obj = rigid_obj_mgr.add_object_by_template_id(new_tempid)
    obj.translation = position
    obj.motion_type = habitat_sim.physics.MotionType.STATIC
    # recompute the NavMesh with STATIC objects
    if navmesh_recompute:
        recompute_navmesh(_sim)
    return obj.object_id

# Remove object from the scene: object_id req.
def remove_object(env, obj_id):
    _sim = env.habitat_env.sim
    rigid_obj_mgr = _sim.get_rigid_object_manager()
    rigid_obj_mgr.remove_object_by_id(obj_id)
    recompute_navmesh(_sim)




