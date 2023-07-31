    # """
    # If human-rendering is used, `self.window` will be a reference
    # to the window that we draw to. `self.clock` will be a clock that is used
    # to ensure that the environment is rendered at the correct framerate in
    # human-mode. They will remain `None` until human-mode is used for the
    # first time.
    # """
    # assert render_mode is None or render_mode in self.metadata['render_modes']
    # self.render_mode = render_mode
    # # The size of the PyGame window
    # self.render_bound = 100
    # self.window_size = 512
    # self.window = None
    # self.clock = None
    #
    # def __render_frame(self, window_size=None, render_bound=None):
    #     if self.window is None and self.render_mode == 'human':
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #
    #     if self.clock is None and self.render_mode == 'human':
    #         self.clock = pygame.time.Clock()
    #
    #     render_bound = self.render_bound if render_bound is None else render_bound
    #     window_size = self.window_size if window_size is None else window_size
    #
    #     canvas = pygame.Surface((window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #
    #     # The size of a single grid square in pixels
    #     pix_square_size = (window_size / render_bound)
    #
    #     leader_color = (255, 0, 0)
    #     follower_color = (0, 0, 255)
    #     obs_poi_color = (0, 255, 0)
    #     non_obs_poi_color = (0, 0, 0)
    #     line_color = (192, 192, 192)
    #
    #     # draw some gridlines
    #     for x in range(render_bound + 1):
    #         pygame.draw.line(
    #             canvas, line_color, (0, pix_square_size * x), (window_size, pix_square_size * x), width=1,
    #         )
    #         pygame.draw.line(
    #             canvas, line_color, (pix_square_size * x, 0), (pix_square_size * x, window_size), width=1,
    #         )
    #
    #     for name, agent in self.leaders.items():
    #         location = np.array(agent.location)
    #         pygame.draw.rect(
    #             canvas, leader_color, pygame.Rect(pix_square_size * location, (pix_square_size, pix_square_size))
    #         )
    #
    #     for name, agent in self.followers.items():
    #         location = np.array(agent.location)
    #         pygame.draw.circle(canvas, follower_color, (location + 0.5) * pix_square_size, pix_square_size / 1.5)
    #
    #     for name, agent in self.pois.items():
    #         # different colors to distinguish if the poi is captured
    #         location = np.array(agent.location)
    #         agent_color = obs_poi_color if agent.observed else non_obs_poi_color
    #         # todo  draw circle around poi indicating the observation radius
    #         pygame.draw.circle(canvas, agent_color, (location + 0.5) * pix_square_size, pix_square_size / 1.5)
    #
    #     if self.render_mode == 'human':
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()
    #
    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to keep the framerate stable.
    #         self.clock.tick(self.metadata['render_fps'])
    #     else:  # rgb_array
    #         np_frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    #         return np_frame
    #     return
    #
    # def __render_rgb(self):
    #     # todo set based on min/max agent locations
    #     render_resolution = (512, 512)
    #     render_bounds = (-5, 55)
    #     scaling = np.divide(render_resolution, render_bounds[1] - render_bounds[0])
    #
    #     agent_colors = {AgentType.Learner: [255, 0, 0], AgentType.Actor: [0, 255, 0], AgentType.Static: [0, 0, 255]}
    #     agent_sizes = {AgentType.Learner: 2, AgentType.Actor: 1, AgentType.Static: 3}
    #
    #     background_color = [255, 255, 255]
    #     line_color = [0, 0, 0]
    #     default_color = [128, 128, 128]
    #     default_size = 2
    #     num_lines = 10
    #     x_line_idxs = np.linspace(0, render_resolution[1], num=num_lines)
    #     y_line_idxs = np.linspace(0, render_resolution[0], num=num_lines)
    #
    #     frame = np.full((render_resolution[0] + 1, render_resolution[1] + 1, 3), background_color)
    #
    #     # draw a grid over the frame
    #     for each_line in x_line_idxs:
    #         each_line = int(each_line)
    #         frame[each_line] = line_color
    #
    #     for each_line in y_line_idxs:
    #         each_line = int(each_line)
    #         frame[:, each_line] = line_color
    #
    #     # place the agents in the frame based on the sizes and colors specified in agent_colors and agent_sizes
    #     for agent_name in self.agents:
    #         agent = self.agent_mapping[agent_name]
    #         acolor = agent_colors.get(agent.type, default_color)
    #         asize = agent_sizes.get(agent.type, default_size)
    #         aloc = np.array(agent.location)
    #
    #         scaled_loc = aloc - render_bounds[0]
    #         scaled_loc = np.multiply(scaled_loc, scaling)
    #         scaled_loc = np.rint(scaled_loc)
    #         scaled_loc = scaled_loc.astype(np.int)
    #         frame[
    #         scaled_loc[1] - asize: scaled_loc[1] + asize,
    #         scaled_loc[0] - asize: scaled_loc[0] + asize,
    #         ] = acolor
    #     frame = frame.astype(np.uint8)
    #     return frame
    #
    # def render(self, mode: str | None = None, **kwargs):
    #     """
    #     Displays a rendered frame from the environment, if supported.
    #
    #     Alternate render modes in the default environments are ‘rgb_array’ which returns a numpy array and
    #     is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed
    #     (specific to classic environments).
    #
    #     :param mode:
    #     :param kwargs:
    #     :return:
    #     """
    #     if not mode:
    #         mode = self.render_mode
    #
    #     match mode:
    #         case 'human':
    #             frame = self.__render_frame(**kwargs)
    #         case 'rgb_array':
    #             frame = self.__render_frame(**kwargs)
    #             # frame = self.__render_rgb()
    #         case _:
    #             frame = None
    #     return frame
    #
    # def close(self):
    #     """
    #     Close should release any graphical displays, subprocesses, network connections or any other
    #     environment data which should not be kept around after the user is no longer using the environment.
    #     """
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
    #     return